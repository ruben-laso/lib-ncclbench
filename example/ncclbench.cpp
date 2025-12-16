#include <iostream>
#include <string>
#include <vector>

#include <ncclbench/ncclbench.hpp>

#include <mpi.h>

#include "include/Options.hpp"
#include "include/Results.hpp"

inline auto generate_cfgs(const Options &options) {
    const auto num_cfgs = options.sizes_bytes.size();

    std::vector<ncclbench::Config> cfgs(num_cfgs);

    auto comm = options.reuse_comm
                    ? std::optional<ncclComm_t>{ncclbench::State::nccl_comm()}
                    : std::optional<ncclComm_t>{std::nullopt};

    for (size_t i = 0; i < num_cfgs; i++) {
        auto &cfg = cfgs[i];
        cfg.operation = options.operation;
        cfg.data_type = options.data_type;
        cfg.bytes_total = options.sizes_bytes[i];
        cfg.blocking = options.blocking;
        cfg.group = options.group;
        cfg.comm = comm;
        // Handle warmup iterations or time
        if (not options.warmup_its.empty() and options.warmup_its[i] > 0) {
            cfg.warmup_its = options.warmup_its[i];
        }
        if (not options.warmup_secs.empty() and options.warmup_secs[i] > 0.0) {
            cfg.warmup_secs = options.warmup_secs[i];
        }

        // Handle benchmark iterations or time
        if (not options.benchmark_its.empty() and
            options.benchmark_its[i] > 0) {
            cfg.benchmark_its = options.benchmark_its[i];
        }
        if (not options.benchmark_secs.empty() and
            options.benchmark_secs[i] > 0.0) {
            cfg.benchmark_secs = options.benchmark_secs[i];
        }
    }

    return cfgs;
}

auto main(int argc, char *argv[]) -> int {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const auto options = parse_options(argc, argv);

    const auto cfgs = generate_cfgs(options);

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

    for (const auto &cfg : cfgs) {
        MPI_Barrier(MPI_COMM_WORLD);
        const auto res = ncclbench::run(cfg);
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

    if (options.reuse_comm) {
        // Destroy communicator. It is the same for all ranks
        ncclCommDestroy(cfgs[0].comm.value());
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
