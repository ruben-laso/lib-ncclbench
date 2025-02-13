#pragma once

#include <string>
#include <vector>

#include <mpi.h>

#include "CLI11.hpp"

struct Options {
    std::string operation = "ncclAllReduce";
    std::string data_type = "float";
    std::vector<size_t> sizes_bytes;
    std::vector<size_t> benchmark_its = {};
    std::vector<size_t> warmup_its = {};
    std::vector<double> benchmark_secs = {};
    std::vector<double> warmup_secs = {};
    bool blocking = false;
    bool csv = false;
    bool summary = false;
    bool reuse_comm = false;
};

template <typename I, typename T, typename S>
inline auto handle_its_or_time(I &its, T &times, const S &sizes) {
    // If both are empty, nothing to do
    if (its.empty() and times.empty()) {
        return;
    }

    // If its are empty, check times
    if (its.empty()) {
        if (times.size() == 1) {
            times.resize(sizes.size(), times[0]);
        } else {
            if (times.size() != sizes.size()) {
                throw CLI::ValidationError{
                    "Number of times must be 1 or equal to number of sizes"};
            }
        }
    } else {
        if (its.size() == 1) {
            its.resize(sizes.size(), its[0]);
        } else {
            if (its.size() != sizes.size()) {
                throw CLI::ValidationError{"Number of iterations must be 1 or "
                                           "equal to number of sizes"};
            }
        }
    }
}

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
                     "Number of benchmark iterations. 1 (same for all) or as "
                     "many as message sizes. "
                     "E.g.: 100 200 300")
        ->check(CLI::PositiveNumber);
    grp_bench_its_secs
        ->add_option("-t,--time", options.benchmark_secs,
                     "Benchmark time in seconds. 1 (same for all) or as many "
                     "as sizes. E.g.: 1 2 3")
        ->check(CLI::PositiveNumber);
    grp_bench_its_secs->require_option(1, 2);

    // At least one of --warmups or --warmup-time must be used
    auto grp_warmup_its_secs =
        app.add_option_group("Warmup iterations or seconds");
    grp_warmup_its_secs
        ->add_option("-w,--warmups", options.warmup_its,
                     "Number of warmup iterations. 1 (same for all) or as many "
                     "as message sizes. "
                     "E.g.: 10 20 30")
        ->check(CLI::NonNegativeNumber);
    grp_warmup_its_secs
        ->add_option(
            "-W,--warmup-time", options.warmup_secs,
            "Warmup time in seconds. 1 (same for all) or as many as sizes. "
            "E.g.: 1 2 3")
        ->check(CLI::NonNegativeNumber);
    grp_warmup_its_secs->require_option(0, 2);

    try {
        app.parse(argc, argv);

        handle_its_or_time(options.warmup_its, options.warmup_secs,
                           options.sizes_bytes);
        handle_its_or_time(options.benchmark_its, options.benchmark_secs,
                           options.sizes_bytes);
    } catch (const CLI::ParseError &e) {
        // Print error and exit
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            std::cerr << e.what() << '\n';
            std::cerr << app.help() << '\n';
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    return options;
}