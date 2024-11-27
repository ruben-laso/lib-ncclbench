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
    size_t iterations = 1'000;
    size_t warmups = 100;
    bool blocking = true;
    bool csv = false;
};

auto main(int argc, char * argv[]) -> int
{
    MPI_Init(&argc, &argv);

    CLI::App app{"NCCL Bench example"};

    Options options;

    app.add_option("-o,--operation", options.operation, "NCCL operation. E.g. ncclAllReduce");
    app.add_option("-d,--data-type", options.data_type, "Data type: [char, int, float, double]");
    app.add_option("-s,--size", options.sizes_bytes, "Size in bytes")->expected(-1);
    app.add_option("-i,--iterations", options.iterations, "Number of iterations");
    app.add_option("-w,--warmups", options.warmups, "Number of warmups");
    app.add_flag("-b,--blocking", options.blocking, "Blocking or non-blocking");
    app.add_flag("--csv", options.csv, "Output in CSV format");

    CLI11_PARSE(app, argc, argv);

    ncclbench::Config config;
    config.operation = options.operation;
    config.data_type = options.data_type;
    config.iterations = options.iterations;
    config.warmups = options.warmups;
    config.blocking = options.blocking;

    std::cout << ncclbench::Results::header();
    for (const auto size : options.sizes_bytes) {
        config.bytes_total = size;
        const auto res = ncclbench::run(config);
        std::cout << (options.csv ? res.csv() : res.text());
    }

    MPI_Finalize();

	return 0;
}
