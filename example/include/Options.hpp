#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <mpi.h>

#include "CLI11.hpp"

struct Options {
    std::filesystem::path config_file = "";
    std::string operation = "ncclAllReduce";
    std::string data_type = "float";
    std::vector<size_t> sizes_bytes;
    std::vector<size_t> benchmark_its = {};
    std::vector<size_t> warmup_its = {};
    std::vector<double> benchmark_secs = {};
    std::vector<double> warmup_secs = {};
    bool blocking = false;
    bool group = false;
    bool csv = false;
    bool summary = false;
    bool reuse_comm = false;
};

template <typename I, typename T, typename S>
inline auto handle_its_and_time(I &its, T &times, const S &sizes) {
    // Resize its
    if (not its.empty()) {
        if (its.size() == 1) {
            its.resize(sizes.size(), its[0]);
        } else {
            if (its.size() != sizes.size()) {
                throw CLI::ValidationError{"Number of iterations must be 1 or "
                                           "equal to number of sizes"};
            }
        }
    }

    // Resize times
    if (not times.empty()) {
        if (times.size() == 1) {
            times.resize(sizes.size(), times[0]);
        } else {
            if (times.size() != sizes.size()) {
                throw CLI::ValidationError{
                    "Number of times must be 1 or equal to number of sizes"};
            }
        }
    }
}

inline auto parse_options(const int argc, char *argv[]) -> Options {
    CLI::App app{"NCCL Bench example"};

    Options options;

    const auto concat_values = [](const std::string &out_str,
                                  const std::string_view &val) {
        return out_str.empty() ? std::string(val)
                               : out_str + " " + std::string(val);
    };

    const std::string operations =
        std::accumulate(std::begin(ncclbench::SUPPORTED_OPERATIONS),
                        std::end(ncclbench::SUPPORTED_OPERATIONS),
                        std::string{}, concat_values);

    const std::string data_types =
        std::accumulate(std::begin(ncclbench::SUPPORTED_DATA_TYPES),
                        std::end(ncclbench::SUPPORTED_DATA_TYPES),
                        std::string{}, concat_values);

    // Configuration file option
    auto grp_config = app.add_option_group("Configuration file (YAML)");
    grp_config
        ->add_option("-c,--config", options.config_file,
                     "YAML configuration file for benchmark settings")
        ->check(CLI::ExistingFile);

    // If no config file is provided, require operation, data type, and sizes
    // Benchmark settings group (config or individual settings)
    auto grp_bench_settings =
        app.add_option_group("Benchmark settings (required if no config file)");
    grp_bench_settings->add_option("-o,--operation", options.operation,
                                   "NCCL operation. Select from: [" +
                                       operations + "]");
    grp_bench_settings->add_option("-d,--data-type", options.data_type,
                                   "Data type. Select from: [" + data_types +
                                       "]");
    grp_bench_settings
        ->add_option("-s,--sizes", options.sizes_bytes,
                     "Size(s) in bytes. E.g.: 1024 2048 4096")
        ->check(CLI::NonNegativeNumber);

    // Either config file or individual settings must be provided
    grp_config->excludes(grp_bench_settings);
    grp_bench_settings->excludes(grp_config);

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
    grp_bench_its_secs->require_option(0, 2);

    // Output options
    auto grp_output = app.add_option_group("Output options");
    grp_output->add_flag("--csv", options.csv, "Output in CSV format");
    grp_output->add_flag("-S,--summary", options.summary, "Print summary");

    // Global options (can be overridden by config file settings)
    auto grp_global = app.add_option_group("Global options");
    grp_global->add_flag("-b,--blocking", options.blocking,
                                 "Blocking or non-blocking");
    grp_global->add_flag(
        "-g,--group", options.group,
        "Enable ncclGroupStart/End (only for non-blocking)");
    grp_global->add_flag("-r,--reuse-comm", options.reuse_comm,
                                 "Reuse NCCL communicator");

    try {
        app.parse(argc, argv);

        handle_its_and_time(options.warmup_its, options.warmup_secs,
                            options.sizes_bytes);
        handle_its_and_time(options.benchmark_its, options.benchmark_secs,
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