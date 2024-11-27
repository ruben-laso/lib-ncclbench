#pragma once

#include <mpi.h>

#include "ncclbench/ncclbench.hpp"

#include "ncclbench/utils/checks.hpp"
#include "ncclbench/utils/types.hpp"
#include "ncclbench/utils/utils.hpp"

namespace ncclbench::benchmark {

namespace utils {
template <typename BwFactor>
auto gather_results(const Config &cfg, const Sizes &sizes,
                    const double local_avg_time, BwFactor &bw_factor)
    -> Results {

    double min_time, max_time, avg_time;
    MPICHECK(MPI_Reduce(&local_avg_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0,
                        MPI_COMM_WORLD));
    MPICHECK(MPI_Reduce(&local_avg_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0,
                        MPI_COMM_WORLD));
    MPICHECK(MPI_Reduce(&local_avg_time, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0,
                        MPI_COMM_WORLD));

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
        r.iterations = cfg.iterations;
        r.warmups = cfg.warmups;
        r.time_min = min_time;
        r.time_max = max_time;
        r.time_avg = avg_time;
        r.bw_alg = alg_bw;
        r.bw_bus = bus_bw;

        return r;
    }

    return {};
}

static void sync_stream(const cudaStream_t stream) {
    CUDACHECK(cudaStreamSynchronize(stream));
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
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

    auto benchmark_loop = [&](const size_t iter) {
        auto comm = State::nccl_comm();
        for (size_t i = 0; i < iter; i++) {
            ncclFunction(buffer_send, buffer_recv, sizes.elements_send,
                         nccl_datatype, comm, stream);
            if (cfg.blocking) {
                utils::sync_stream(stream);
            }
        }
        if (not cfg.blocking) {
            utils::sync_stream(stream);
        }
    };

    // Warmup
    MPICHECK(MPI_Barrier(State::mpi_comm()));
    benchmark_loop(cfg.warmups);

    // Benchmark
    MPICHECK(MPI_Barrier(State::mpi_comm()));
    const double start = MPI_Wtime();
    benchmark_loop(cfg.iterations);
    const double end = MPI_Wtime();
    const double avg_time = (end - start) / static_cast<double>(cfg.iterations);

    // Report performance metrics
    const auto results = utils::gather_results(cfg, sizes, avg_time, bw_factor);

    // Cleanup
    CUDACHECK(cudaStreamDestroy(stream));
    CUDACHECK(cudaFree(buffer_send));
    CUDACHECK(cudaFree(buffer_recv));
    free(buffer_host);

    return results;
}
} // namespace ncclbench::benchmark
