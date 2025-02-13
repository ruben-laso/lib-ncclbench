#include <iostream>
#include <string>
#include <vector>

#include <ncclbench/ncclbench.hpp>

#include <mpi.h>

#include "include/Options.hpp"
#include "include/Results.hpp"

auto main(int argc, char *argv[]) -> int {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const auto options = parse_options(argc, argv);

    ncclbench::Config config;
    config.operation = options.operation;
    config.data_type = options.data_type;
    config.blocking = options.blocking;
    if (options.reuse_comm) {
        config.comm = ncclbench::State::nccl_comm();
    } else {
        config.comm = std::nullopt;
    }

    if (options.warmup_its > 0) {
        config.warmup_its = options.warmup_its;
    }
    if (options.warmup_secs > 0.0) {
        config.warmup_secs = options.warmup_secs;
    }
    if (not config.warmup_its.has_value() and
        not config.warmup_secs.has_value()) {
        config.warmup_its = 0;
    }

    if (options.benchmark_its > 0) {
        config.benchmark_its = options.benchmark_its;
    }
    if (options.benchmark_secs > 0.0) {
        config.benchmark_secs = options.benchmark_secs;
    }

    if (rank == 0) {
        if (options.summary) {
            std::cout << (options.csv ? ResultSummary::csv_header()
                                      : ResultSummary::header())
                      << std::endl;
        } else {
            std::cout << (options.csv ? ncclbench::Result::csv_header()
                                      : ncclbench::Result::header())
                      << std::endl;
        }
    }

    for (const auto size : options.sizes_bytes) {
        config.bytes_total = size;
        MPI_Barrier(MPI_COMM_WORLD);
        const auto res = ncclbench::run(config);
        if (rank == 0) {
            if (options.summary) {
                ResultSummary summary(res);
                std::cout << (options.csv ? summary.csv() : summary.text())
                          << std::endl;
            } else {
                for (const auto &r : res) {
                    std::cout << (options.csv ? r.csv() : r.text()) << '\n';
                }
            }
        }
    }

    if (config.comm.has_value()) {
        ncclCommDestroy(config.comm.value());
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
