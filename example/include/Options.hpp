#pragma once

#include <string>
#include <vector>

#include <mpi.h>

#include "CLI11.hpp"

struct Options {
    std::string operation = "ncclAllReduce";
    std::string data_type = "float";
    std::vector<size_t> sizes_bytes;
    size_t benchmark_its = 0;
    size_t warmup_its = 0;
    double benchmark_secs = 0.0;
    double warmup_secs = 0.0;
    bool blocking = false;
    bool csv = false;
    bool summary = false;
    bool reuse_comm = false;
};

inline auto parse_options(const int argc, char *argv[]) -> Options {
    CLI::App app{"NCCL Bench example"};

    Options options;

    const std::string operations =
        std::accumulate(ncclbench::SUPPORTED_OPERATIONS.begin(),
                        ncclbench::SUPPORTED_OPERATIONS.end(), std::string{},
                        [](const std::string &a, const std::string &b) {
                            return a.empty() ? b : a + " " + b;
                        });

    const std::string data_types =
        std::accumulate(ncclbench::SUPPORTED_DATA_TYPES.begin(),
                        ncclbench::SUPPORTED_DATA_TYPES.end(), std::string{},
                        [](const std::string &a, const std::string &b) {
                            return a.empty() ? b : a + " " + b;
                        });

    app.add_option("-o,--operation", options.operation,
                   "NCCL operation. Select from: [" + operations + "]")
        ->required();
    app.add_option("-d,--data-type", options.data_type,
                   "Data type. Select from: [" + data_types + "]")
        ->required();
    app.add_option("-s,--sizes", options.sizes_bytes,
                   "Size(s) in bytes. E.g.: 1024 2048 4096")
        ->required()
        ->check(CLI::NonNegativeNumber);
    app.add_flag("-b,--blocking", options.blocking, "Blocking or non-blocking");
    app.add_flag("-r,--reuse-comm", options.reuse_comm,
                 "Reuse NCCL communicator");
    app.add_flag("--csv", options.csv, "Output in CSV format");
    app.add_flag("-S,--summary", options.summary, "Print summary");

    // At least one of --iterations or --time must be used
    auto grp_bench_its_secs =
        app.add_option_group("Bench. iterations or seconds");
    grp_bench_its_secs
        ->add_option("-i,--iterations", options.benchmark_its,
                     "Number of benchmark iterations")
        ->check(CLI::PositiveNumber);
    grp_bench_its_secs
        ->add_option("-t,--time", options.benchmark_secs,
                     "Benchmark time in seconds")
        ->check(CLI::PositiveNumber);
    grp_bench_its_secs->require_option(1, 2);

    // At least one of --warmups or --warmup-time must be used
    auto grp_warmup_its_secs =
        app.add_option_group("Warmup iterations or seconds");
    grp_warmup_its_secs
        ->add_option("-w,--warmups", options.warmup_its,
                     "Number of warmup iterations")
        ->check(CLI::NonNegativeNumber);
    grp_warmup_its_secs
        ->add_option("-W,--warmup-time", options.warmup_secs,
                     "Warmup time in seconds")
        ->check(CLI::NonNegativeNumber);
    grp_warmup_its_secs->require_option(1, 2);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        // show help in process 0
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            std::cerr << app.help() << '\n';
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    return options;
}