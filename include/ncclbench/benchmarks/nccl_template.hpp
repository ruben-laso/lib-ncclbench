#pragma once

#include <mpi.h>

#include "ncclbench/ncclbench.hpp"

#include "ncclbench/utils/checks.hpp"
#include "ncclbench/utils/types.hpp"
#include "ncclbench/utils/utils.hpp"

#include <iostream>
#include <limits>

namespace ncclbench::benchmark {

namespace utils {

inline auto get_nccl_id() -> ncclUniqueId {
    ncclUniqueId id;
    if (State::rank() == 0) {
        NCCLCHECK(ncclGetUniqueId(&id));
    }
    MPICHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, State::mpi_comm()));
    return id;
}

inline auto get_nccl_comm() -> ncclComm_t {
    const auto nccl_id = get_nccl_id();
    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, State::ranks(), nccl_id, State::rank()));
    return comm;
}

static void sync_stream(cudaStream_t stream) {
    CUDACHECK(cudaStreamSynchronize(stream));
    MPICHECK(MPI_Barrier(State::mpi_comm()));
}

namespace blocking {
template <typename BwFactor>
auto gather_results(const Config &cfg, const Sizes &sizes,
                    std::vector<double> &local_times, BwFactor &&bw_factor)
    -> std::vector<Result> {
    std::vector<Result> results(local_times.size());

    std::vector<double> mins(local_times.size());
    std::vector<double> maxs(local_times.size());
    std::vector<double> avgs(local_times.size());

    MPICHECK(MPI_Reduce(local_times.data(), mins.data(), local_times.size(),
                        MPI_DOUBLE, MPI_MIN, 0, State::mpi_comm()));
    MPICHECK(MPI_Reduce(local_times.data(), maxs.data(), local_times.size(),
                        MPI_DOUBLE, MPI_MAX, 0, State::mpi_comm()));
    MPICHECK(MPI_Reduce(local_times.data(), avgs.data(), local_times.size(),
                        MPI_DOUBLE, MPI_SUM, 0, State::mpi_comm()));

    for (size_t i = 0; i < local_times.size(); i++) {
        avgs[i] /= State::ranks();
    }

    for (size_t i = 0; i < local_times.size(); i++) {
        const auto alg_bw =
            ncclbench::utils::to_GB(sizes.bytes_total) / maxs[i];
        const auto bus_bw = alg_bw * bw_factor();

        results[i] = {cfg.operation,
                      cfg.blocking,
                      cfg.data_type,
                      sizes.bytes_total,
                      sizes.elements_per_rank,
                      1,
                      mins[i],
                      maxs[i],
                      avgs[i],
                      alg_bw,
                      bus_bw};
    }

    return results;
}

template <typename NCCLCall, typename BwFactor>
auto benchmark_loop(const Config &cfg, const Sizes &sizes, NCCLCall &&nccl_call,
                    BwFactor &&bw_factor, cudaStream_t &stream)
    -> std::vector<Result> {
    const auto max_its = cfg.benchmark_its.has_value()
                             ? cfg.benchmark_its.value()
                             : std::numeric_limits<size_t>::max();

    const auto tgt_time = cfg.benchmark_secs.has_value()
                              ? cfg.benchmark_secs.value()
                              : std::numeric_limits<double>::infinity();

    std::vector<double> local_times;

    double accum_time = 0.0;
    size_t its = 0;
    bool stop = false;
    for (its = 0; its < max_its and not stop; its++) {
        const auto start = MPI_Wtime();
        nccl_call();
        sync_stream(stream);
        const auto end = MPI_Wtime();
        const auto elapsed = end - start;
        local_times.push_back(elapsed);
        accum_time += elapsed;
        if (cfg.benchmark_secs.has_value()) {
            stop = accum_time >= tgt_time;
            MPICHECK(MPI_Bcast(&stop, 1, MPI_C_BOOL, 0, State::mpi_comm()));
        }
    }

    return gather_results(cfg, sizes, local_times, bw_factor);
}

} // namespace blocking

namespace nonblocking {
template <typename BwFactor>
auto gather_results(const Config &cfg, const Sizes &sizes, double local_time,
                    BwFactor &&bw_factor) -> std::vector<Result> {
    assert(not cfg.blocking);

    std::vector<Result> results(1);

    double min;
    double max;
    double avg;

    MPICHECK(MPI_Reduce(&local_time, &min, 1, MPI_DOUBLE, MPI_MIN, 0,
                        State::mpi_comm()));
    MPICHECK(MPI_Reduce(&local_time, &max, 1, MPI_DOUBLE, MPI_MAX, 0,
                        State::mpi_comm()));
    MPICHECK(MPI_Reduce(&local_time, &avg, 1, MPI_DOUBLE, MPI_SUM, 0,
                        State::mpi_comm()));

    avg /= State::ranks();

    const auto alg_bw = ncclbench::utils::to_GB(sizes.bytes_total) / max;
    const auto bus_bw = alg_bw * bw_factor();

    results[0] = {cfg.operation,
                  cfg.blocking,
                  cfg.data_type,
                  sizes.bytes_total,
                  sizes.elements_per_rank,
                  cfg.benchmark_its.value(),
                  min,
                  max,
                  avg,
                  alg_bw,
                  bus_bw};

    return results;
}

template <typename NCCLCall, typename BwFactor>
auto benchmark_loop(const Config &cfg, const Sizes &sizes, NCCLCall &&nccl_call,
                    BwFactor &&bw_factor, cudaStream_t &stream)
    -> std::vector<Result> {
    if (not cfg.benchmark_its.has_value()) {
        throw std::runtime_error(
            "Time-based benchmarking requires blocking operations");
    }

    if (cfg.benchmark_secs.has_value()) {
        std::cerr << __FILE__ << ":" << __LINE__ << '\t'
                  << "Warning: Ignoring time limit when benchmarking "
                     "non-blocking operations"
                  << std::endl;
    }

    const auto max_its = cfg.benchmark_its.value();

    const auto start = MPI_Wtime();
    for (size_t i = 0; i < max_its; i++) {
        nccl_call();
    }
    const auto end = MPI_Wtime();

    return gather_results(cfg, sizes, end - start, bw_factor);
}

} // namespace nonblocking

template <typename NCCLCall, typename BwFactor>
auto benchmark_loop(const Config &cfg, const Sizes &sizes, NCCLCall &&nccl_call,
                    BwFactor &&bw_factor, cudaStream_t &stream)
    -> std::vector<Result> {
    if (not cfg.benchmark_its.has_value() and
        not cfg.benchmark_secs.has_value()) {
        // Something went wrong
        throw std::runtime_error("Invalid benchmarking configuration. Either "
                                 "its or secs must be set");
    }

    return cfg.blocking ? blocking::benchmark_loop(cfg, sizes, nccl_call,
                                                   bw_factor, stream)
                        : nonblocking::benchmark_loop(cfg, sizes, nccl_call,
                                                      bw_factor, stream);
}

} // namespace utils

template <typename NCCLFunction, typename BwFactor>
auto run_benchmark(const Config &cfg, const Sizes &sizes,
                   NCCLFunction &&ncclFunction, BwFactor &&bw_factor)
    -> std::vector<Result> {

    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));

    // Allocate memory
    void *buffer_send;
    void *buffer_recv;
    CUDACHECK(cudaMalloc(&buffer_send, sizes.bytes_send));
    CUDACHECK(cudaMalloc(&buffer_recv, sizes.bytes_recv));
    void *buffer_host = malloc(sizes.bytes_send);
    ncclbench::utils::init_data(buffer_host, types::str_to_mpi(cfg.data_type),
                                sizes.elements_send);
    CUDACHECK(cudaMemcpy(buffer_send, buffer_host, sizes.bytes_send,
                         cudaMemcpyHostToDevice));
    CUDACHECK(cudaDeviceSynchronize());

    // Benchmark function
    const auto nccl_datatype = types::str_to_nccl(cfg.data_type);
    const auto comm = utils::get_nccl_comm();
    auto nccl_call = [&]() {
        ncclFunction(buffer_send, buffer_recv, sizes.elements_per_rank,
                     nccl_datatype, comm, stream);
    };

    // Warmup
    {
        MPICHECK(MPI_Barrier(State::mpi_comm()));
        auto warmup_cfg = cfg;
        warmup_cfg.benchmark_its = cfg.warmup_its;
        warmup_cfg.benchmark_secs = cfg.warmup_secs;
        std::ignore = utils::benchmark_loop(warmup_cfg, sizes, nccl_call,
                                            bw_factor, stream);
    }

    // Benchmark
    MPICHECK(MPI_Barrier(State::mpi_comm()));
    const auto results =
        utils::benchmark_loop(cfg, sizes, nccl_call, bw_factor, stream);

    // Cleanup
    NCCLCHECK(ncclCommDestroy(comm));
    CUDACHECK(cudaStreamDestroy(stream));
    CUDACHECK(cudaFree(buffer_send));
    CUDACHECK(cudaFree(buffer_recv));
    free(buffer_host);

    return results;
}
} // namespace ncclbench::benchmark
