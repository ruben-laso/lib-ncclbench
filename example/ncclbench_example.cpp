#include <iostream>
#include <string>
#include <vector>

#include <ncclbench/ncclbench.hpp>

#include "include/CLI11.hpp"

#include <mpi.h>

struct Options
{
    std::string operation = "ncclAllReduce";
    std::string data_type = "float";
    std::vector<size_t> sizes_bytes;
    size_t benchmark_its = 0;
    size_t warmup_its = 0;
    double benchmark_secs = 0.0;
    double warmup_secs = 0.0;
    bool blocking = false;
    bool csv = false;
};

auto main(int argc, char * argv[]) -> int
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    CLI::App app{"NCCL Bench example"};

    Options options;

    app.add_option("-o,--operation", options.operation, "NCCL operation. E.g. ncclAllReduce")->required();
    app.add_option("-d,--data-type", options.data_type, "Data type: [byte, char, int, float, double]")->required();
    app.add_option("-s,--sizes", options.sizes_bytes, "Size(s) in bytes. E.g.: 1024 2048 4096")->required()->check(CLI::NonNegativeNumber);
    app.add_flag("-b,--blocking", options.blocking, "Blocking or non-blocking");
    app.add_flag("--csv", options.csv, "Output in CSV format");

    // Only one of --iterations or --time can be used
    auto grp_bench_its_secs = app.add_option_group("Bench. iterations or seconds");
    grp_bench_its_secs->add_option("-i,--iterations", options.benchmark_its, "Number of benchmark iterations")->check(CLI::PositiveNumber);
    grp_bench_its_secs->add_option("-t,--time", options.benchmark_secs, "Benchmark time in seconds")->check(CLI::PositiveNumber);
    grp_bench_its_secs->require_option(1);

    // Only one of --warmups or --warmup-time can be used
    auto grp_warmup_its_secs = app.add_option_group("Warmup iterations or seconds");
    grp_warmup_its_secs->add_option("-w,--warmups", options.warmup_its, "Number of warmup iterations")->check(CLI::NonNegativeNumber);
    grp_warmup_its_secs->add_option("-W,--warmup-time", options.warmup_secs, "Warmup time in seconds")->check(CLI::NonNegativeNumber);
    grp_warmup_its_secs->require_option(1);

    CLI11_PARSE(app, argc, argv);

    ncclbench::Config config;
    config.operation = options.operation;
    config.data_type = options.data_type;
    config.blocking = options.blocking;
    if (options.warmup_its == 0) { config.warmup_its_or_secs = options.warmup_secs; }
    else { config.warmup_its_or_secs = options.warmup_its; }
    if (options.benchmark_its == 0) { config.benchmark_its_or_secs = options.benchmark_secs; }
    else { config.benchmark_its_or_secs = options.benchmark_its; }

    if (rank == 0) {
        std::cout << (options.csv ? ncclbench::Results::csv_header() : ncclbench::Results::header()) << std::endl;
    }

    for (const auto size : options.sizes_bytes) {
        config.bytes_total = size;
        MPI_Barrier(MPI_COMM_WORLD);
        const auto res = ncclbench::run(config);
        if (rank == 0) {
        	std::cout << (options.csv ? res.csv() : res.text()) << std::endl;
        }
    }

    MPI_Finalize();

	return EXIT_SUCCESS;
}
