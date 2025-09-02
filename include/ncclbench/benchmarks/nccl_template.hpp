#pragma once

#include <mpi.h>

#include "ncclbench/ncclbench.hpp"

#include "ncclbench/utils/checks.hpp"
#include "ncclbench/utils/types.hpp"
#include "ncclbench/utils/utils.hpp"

#include <iostream>
#include <limits>
#include <tuple>
#include <utility>

namespace ncclbench::benchmark {

namespace utils {
static void sync_stream(cudaStream_t stream) {
    CUDACHECK(cudaStreamSynchronize(stream));
}

namespace blocking {
template <typename BwFactor>
auto gather_results(const Config &cfg, const Sizes &sizes,
                    std::vector<double> &local_begins,
                    std::vector<double> &local_ends, BwFactor &&bw_factor)
    -> std::vector<Result> {
    std::vector<Result> results(local_begins.size());

    std::vector<double> begins(local_begins.size());
    std::vector<double> ends(local_begins.size());

    MPICHECK(MPI_Reduce(local_begins.data(), begins.data(), local_begins.size(),
                        MPI_DOUBLE, MPI_MIN, 0, State::mpi_comm()));
    MPICHECK(MPI_Reduce(local_ends.data(), ends.data(), local_ends.size(),
                        MPI_DOUBLE, MPI_MAX, 0, State::mpi_comm()));

    for (size_t i = 0; i < local_begins.size(); i++) {
        const auto wall_time = ends[i] - begins[i];
        const auto alg_bw =
            ncclbench::utils::to_GB(sizes.bytes_total) / wall_time;
        const auto bus_bw = alg_bw * bw_factor();

        results[i] = {cfg.operation,
                      cfg.blocking,
                      cfg.data_type,
                      sizes.bytes_total,
                      sizes.elements_per_rank,
                      1,
                      wall_time,
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

    std::vector<double> local_begins;
    std::vector<double> local_ends;
    if (max_its < std::numeric_limits<size_t>::max()) {
        local_begins.reserve(max_its);
        local_ends.reserve(max_its);
    }

    double accum_time = 0.0;
    size_t its = 0;
    bool stop = false;
    for (its = 0; its < max_its and not stop; its++) {
        MPICHECK(MPI_Barrier(State::mpi_comm()));
        const auto begin = MPI_Wtime();
        nccl_call();
        sync_stream(stream);
        const auto end = MPI_Wtime();
        const auto elapsed = end - begin;
        local_begins.push_back(begin);
        local_ends.push_back(end);
        accum_time += elapsed;
        if (cfg.benchmark_secs.has_value()) {
            stop = accum_time >= tgt_time;
            MPICHECK(MPI_Bcast(&stop, 1, MPI_C_BOOL, 0, State::mpi_comm()));
        }
    }

    return gather_results(cfg, sizes, local_begins, local_ends, bw_factor);
}
} // namespace blocking

namespace nonblocking {
template <typename BwFactor>
auto gather_results(const Config &cfg, const Sizes &sizes, double local_begin,
                    double local_end, BwFactor &&bw_factor)
    -> std::vector<Result> {
    assert(not cfg.blocking);

    std::vector<Result> results(1);

    double begin;
    double end;

    MPICHECK(MPI_Reduce(&local_begin, &begin, 1, MPI_DOUBLE, MPI_MIN, 0,
                        State::mpi_comm()));
    MPICHECK(MPI_Reduce(&local_end, &end, 1, MPI_DOUBLE, MPI_MAX, 0,
                        State::mpi_comm()));

    const auto wall_time = (end - begin) / cfg.benchmark_its.value();
    const auto alg_bw = ncclbench::utils::to_GB(sizes.bytes_total) / wall_time;
    const auto bus_bw = alg_bw * bw_factor();

    results[0] = {cfg.operation,
                  cfg.blocking,
                  cfg.data_type,
                  sizes.bytes_total,
                  sizes.elements_per_rank,
                  cfg.benchmark_its.value(),
                  wall_time,
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

    const auto begin = MPI_Wtime();
    for (size_t i = 0; i < max_its; i++) {
        nccl_call();
    }
    const auto end = MPI_Wtime();

    return gather_results(cfg, sizes, begin, end, bw_factor);
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
    ncclbench::utils::init_data(buffer_host, types::str_to_nccl(cfg.data_type),
                                sizes.elements_send);
    CUDACHECK(cudaMemcpy(buffer_send, buffer_host, sizes.bytes_send,
                         cudaMemcpyHostToDevice));
    CUDACHECK(cudaDeviceSynchronize());

    // Benchmark function
    const auto nccl_datatype = types::str_to_nccl(cfg.data_type);
    const auto comm =
        cfg.comm.has_value() ? cfg.comm.value() : State::nccl_comm();
    auto nccl_call = [&]() {
        ncclFunction(buffer_send, buffer_recv, sizes.elements_per_rank,
                     nccl_datatype, comm, stream);
    };

    // Warmup
    if (cfg.warmup_its.has_value() or cfg.warmup_secs.has_value()) {
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
    if (not cfg.comm.has_value()) {
        // Remove temporary communicator
        NCCLCHECK(ncclCommDestroy(comm));
    }
    CUDACHECK(cudaStreamDestroy(stream));
    CUDACHECK(cudaFree(buffer_send));
    CUDACHECK(cudaFree(buffer_recv));
    free(buffer_host);

    return results;
}
} // namespace ncclbench::benchmark
