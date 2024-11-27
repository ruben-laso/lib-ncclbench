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

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    CLI::App app{"NCCL Bench example"};

    Options options;

    app.add_option("-o,--operation", options.operation, "NCCL operation. E.g. ncclAllReduce")->required();
    app.add_option("-d,--data-type", options.data_type, "Data type: [byte, char, int, float, double]")->required();
    app.add_option("-s,--sizes", options.sizes_bytes, "Size(s) in bytes. E.g.: 1024 2048 4096")->required();
    app.add_option("-i,--iterations", options.iterations, "Number of iterations");
    app.add_option("-w,--warmups", options.warmups, "Number of warmups");
    app.add_flag("-b,--blocking", options.blocking, "Blocking or non-blocking");
    app.add_flag("--csv", options.csv, "Output in CSV format");

    CLI11_PARSE(app, argc, argv);

    if (rank == 0 and options.sizes_bytes.empty()) {
        std::cerr << "No sizes provided" << std::endl;
        return EXIT_FAILURE;
    }

    ncclbench::Config config;
    config.operation = options.operation;
    config.data_type = options.data_type;
    config.iterations = options.iterations;
    config.warmups = options.warmups;
    config.blocking = options.blocking;

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
