#include <iostream>
#include <string>
#include <vector>

#include <ncclbench/ncclbench.hpp>

#include <mpi.h>

#include "include/Configs.hpp"
#include "include/Options.hpp"
#include "include/Results.hpp"

auto main(int argc, char *argv[]) -> int {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const auto options = parse_options(argc, argv);

    const auto cfgs = cfgs::generate_cfgs(options);

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
