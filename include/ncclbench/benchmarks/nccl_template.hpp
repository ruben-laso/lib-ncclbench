#pragma once

#include <mpi.h>

#include "ncclbench/ncclbench.hpp"

#include "ncclbench/utils/checks.hpp"
#include "ncclbench/utils/types.hpp"
#include "ncclbench/utils/utils.hpp"

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

template <typename BwFactor>
auto gather_results(const Config &cfg, const Sizes &sizes,
                    const double local_avg_time, const size_t warmup_its,
                    const size_t benchmark_its, const BwFactor &bw_factor)
    -> Results {

    double min_time, max_time, avg_time;
    MPICHECK(MPI_Reduce(&local_avg_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0,
                        State::mpi_comm()));
    MPICHECK(MPI_Reduce(&local_avg_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0,
                        State::mpi_comm()));
    MPICHECK(MPI_Reduce(&local_avg_time, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0,
                        State::mpi_comm()));

    if (State::rank() == 0) {
        avg_time /= State::ranks();
        const double bytes_total_GB =
            static_cast<double>(sizes.bytes_total) / 1.0E9;
        const double alg_bw = bytes_total_GB / avg_time; // GB/s
        const double factor = bw_factor();     // Factor for BW calculation
        const double bus_bw = alg_bw * factor; // Bus bandwidth in GB/s

        Results r{};
        r.operation = cfg.operation;
        r.blocking = cfg.blocking;
        r.data_type = cfg.data_type;
        r.bytes_total = sizes.bytes_total;
        r.elements_per_rank = sizes.elements_per_rank;
        r.benchmark_its = benchmark_its;
        r.warmup_its = warmup_its;
        r.time_min = min_time;
        r.time_max = max_time;
        r.time_avg = avg_time;
        r.bw_alg = alg_bw;
        r.bw_bus = bus_bw;

        return r;
    }

    return {};
}

static void sync_stream(cudaStream_t stream) {
    CUDACHECK(cudaStreamSynchronize(stream));
    MPICHECK(MPI_Barrier(State::mpi_comm()));
}

template <typename NCCLCall>
auto benchmark_loop_its(const size_t &its, const bool blocking,
                        cudaStream_t &stream, NCCLCall &&nccl_call)
    -> std::pair<size_t, double> {
    const auto start = MPI_Wtime();
    for (size_t i = 0; i < its; i++) {
        nccl_call();
        if (blocking) {
            sync_stream(stream);
        }
    }
    if (not blocking) {
        sync_stream(stream);
    }
    const auto end = MPI_Wtime();

    return {its, end - start};
}

template <typename NCCLCall>
auto benchmark_loop_time(const double &time, const bool blocking,
                         cudaStream_t &stream, NCCLCall &&nccl_call)
    -> std::pair<size_t, double> {
    if (not blocking) {
        throw std::runtime_error(
            "Time-based benchmarking requires blocking operations");
    }

    size_t its = 0;
    double accum_time = 0.0;
    bool stop = false;
    while (not stop) {
        const auto start = MPI_Wtime();
        nccl_call();
        sync_stream(stream);
        const auto end = MPI_Wtime();
        accum_time += end - start;
        its++;
        stop = accum_time >= time;
        MPICHECK(MPI_Bcast(&stop, 1, MPI_C_BOOL, 0, State::mpi_comm()));
    }

    return {its, accum_time};
}

template <typename NCCLCall>
auto benchmark_loop_its_and_time(const size_t &max_its, const double &max_time,
                                 const bool blocking, cudaStream_t &stream,
                                 NCCLCall &&nccl_call)
    -> std::pair<size_t, double> {
    if (not blocking) {
        throw std::runtime_error(
            "Time-based benchmarking requires blocking operations");
    }

    double accum_time = 0.0;
    size_t its = 0;
    bool stop = false;
    for (its = 0; its < max_its and not stop; its++) {
        const auto start = MPI_Wtime();
        nccl_call();
        sync_stream(stream);
        const auto end = MPI_Wtime();
        accum_time += end - start;
        stop = accum_time >= max_time;
        MPICHECK(MPI_Bcast(&stop, 1, MPI_C_BOOL, 0, State::mpi_comm()));
    }

    return {its, accum_time};
}

template <typename NCCLCall>
auto benchmark_loop(const std::optional<size_t> &its,
                    const std::optional<double> &secs, const bool blocking,
                    cudaStream_t &stream, NCCLCall &&nccl_call)
    -> std::pair<size_t, double> {
    if (its.has_value() and secs.has_value()) {
        return benchmark_loop_its_and_time(its.value(), secs.value(), blocking,
                                           stream, nccl_call);
    }
    if (its.has_value()) {
        return benchmark_loop_its(its.value(), blocking, stream, nccl_call);
    }
    if (secs.has_value()) {
        return benchmark_loop_time(secs.value(), blocking, stream, nccl_call);
    }
    // Something went wrong
    throw std::runtime_error("Invalid benchmarking configuration");
}

} // namespace utils

template <typename NCCLFunction, typename BwFactor>
auto run_benchmark(const Config &cfg, const Sizes &sizes,
                   NCCLFunction &&ncclFunction, BwFactor &&bw_factor)
    -> Results {

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
    MPICHECK(MPI_Barrier(State::mpi_comm()));
    const auto [warmup_its, warmup_time] = utils::benchmark_loop(
        cfg.warmup_its, cfg.warmup_secs, cfg.blocking, stream, nccl_call);

    // Benchmark
    MPICHECK(MPI_Barrier(State::mpi_comm()));
    const auto [bench_its, bench_time] = utils::benchmark_loop(
        cfg.benchmark_its, cfg.benchmark_secs, cfg.blocking, stream, nccl_call);
    const double avg_time = bench_time / static_cast<double>(bench_its);

    // Report performance metrics
    const auto results = utils::gather_results(cfg, sizes, avg_time, warmup_its,
                                               bench_its, bw_factor);

    // Cleanup
    NCCLCHECK(ncclCommDestroy(comm));
    CUDACHECK(cudaStreamDestroy(stream));
    CUDACHECK(cudaFree(buffer_send));
    CUDACHECK(cudaFree(buffer_recv));
    free(buffer_host);

    return results;
}
} // namespace ncclbench::benchmark
