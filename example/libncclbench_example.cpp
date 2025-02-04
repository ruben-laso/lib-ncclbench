#include <iostream>
#include <string>
#include <vector>

#include <ncclbench/ncclbench.hpp>

#include "include/CLI11.hpp"

#include <mpi.h>

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
};

struct Stats {
    double min;
    double max;
    double avg;
    double stddev;
};

struct ResultSummary {
    std::string operation;
    bool blocking;
    std::string data_type;
    size_t bytes_total;
    size_t elements_per_rank;
    size_t benchmark_its;
    Stats time;
    Stats bw_alg;
    Stats bw_bus;

    // Results summary from std::vector<Result>
    ResultSummary(const std::vector<ncclbench::Result> &results) {
        if (results.empty()) {
            throw std::runtime_error{"Empty results"};
        }
        operation = results[0].operation;
        data_type = results[0].data_type;
        bytes_total = results[0].bytes_total;
        elements_per_rank = results[0].elements_per_rank;
        benchmark_its = std::transform_reduce(
            results.begin(), results.end(), 0, std::plus<>{},
            [](const auto &a) { return a.benchmark_its; });

        // Time stats
        time.min = std::min_element(results.begin(), results.end(),
                                    [](const auto &a, const auto &b) {
                                        return a.time_min < b.time_min;
                                    })
                       ->time_min;
        time.max = std::max_element(results.begin(), results.end(),
                                    [](const auto &a, const auto &b) {
                                        return a.time_max < b.time_max;
                                    })
                       ->time_max;
        time.avg = std::transform_reduce(
                       results.begin(), results.end(), 0.0, std::plus<>{},
                       [](const auto &a) { return a.time_avg; }) /
                   results.size();
        time.stddev =
            std::sqrt(std::transform_reduce(
                          results.begin(), results.end(), 0.0, std::plus<>{},
                          [avg = time.avg](const auto &a) {
                              return std::pow(a.time_max - avg, 2);
                          }) /
                      results.size());

        // Bandwidth stats
        bw_alg.min = std::min_element(results.begin(), results.end(),
                                      [](const auto &a, const auto &b) {
                                          return a.bw_alg < b.bw_alg;
                                      })
                         ->bw_alg;
        bw_alg.max = std::max_element(results.begin(), results.end(),
                                      [](const auto &a, const auto &b) {
                                          return a.bw_alg < b.bw_alg;
                                      })
                         ->bw_alg;
        bw_alg.avg = std::transform_reduce(
                         results.begin(), results.end(), 0.0, std::plus<>{},
                         [](const auto &a) { return a.bw_alg; }) /
                     results.size();
        bw_alg.stddev =
            std::sqrt(std::transform_reduce(
                          results.begin(), results.end(), 0.0, std::plus<>{},
                          [avg = bw_alg.avg](const auto &a) {
                              return std::pow(a.bw_alg - avg, 2);
                          }) /
                      results.size());

        // Bus bandwidth stats
        bw_bus.min = std::min_element(results.begin(), results.end(),
                                      [](const auto &a, const auto &b) {
                                          return a.bw_bus < b.bw_bus;
                                      })
                         ->bw_bus;
        bw_bus.max = std::max_element(results.begin(), results.end(),
                                      [](const auto &a, const auto &b) {
                                          return a.bw_bus < b.bw_bus;
                                      })
                         ->bw_bus;
        bw_bus.avg = std::transform_reduce(
                         results.begin(), results.end(), 0.0, std::plus<>{},
                         [](const auto &a) { return a.bw_bus; }) /
                     results.size();
        bw_bus.stddev =
            std::sqrt(std::transform_reduce(
                          results.begin(), results.end(), 0.0, std::plus<>{},
                          [avg = bw_bus.avg](const auto &a) {
                              return std::pow(a.bw_bus - avg, 2);
                          }) /
                      results.size());
    }

    static auto header() -> std::string {
        using namespace ncclbench;

        std::ostringstream oss;

        oss << std::left                                            //
            << std::setw(Result::LRG_WIDTH) << "Operation"          //
            << std::setw(Result::SML_WIDTH) << "Blocking"           //
            << std::setw(Result::MID_WIDTH) << "Data Type"          //
            << std::right                                           //
            << std::setw(Result::LRG_WIDTH) << "Msg Size (B)"       //
            << std::setw(Result::MID_WIDTH) << "#Elements"          //
            << std::setw(Result::MID_WIDTH) << "Iterations"         //
            << std::setw(Result::MID_WIDTH) << "Time min. (us)"     //
            << std::setw(Result::MID_WIDTH) << "Time max. (us)"     //
            << std::setw(Result::MID_WIDTH) << "Time avg. (us)"     //
            << std::setw(Result::MID_WIDTH) << "Time std. (us)"     //
            << std::setw(Result::LRG_WIDTH) << "Alg BW min. (GB/s)" //
            << std::setw(Result::LRG_WIDTH) << "Alg BW max. (GB/s)" //
            << std::setw(Result::LRG_WIDTH) << "Alg BW avg. (GB/s)" //
            << std::setw(Result::LRG_WIDTH) << "Alg BW std. (GB/s)" //
            << std::setw(Result::LRG_WIDTH) << "Bus BW min. (GB/s)" //
            << std::setw(Result::LRG_WIDTH) << "Bus BW max. (GB/s)" //
            << std::setw(Result::LRG_WIDTH) << "Bus BW avg. (GB/s)" //
            << std::setw(Result::LRG_WIDTH) << "Bus BW std. (GB/s)";

        return oss.str();
    }

    static auto csv_header() -> std::string {
        std::ostringstream oss;

        oss << "Operation,"      //
            << "Blocking,"       //
            << "Data_Type,"      //
            << "Msg_Size_B,"     //
            << "#Elements,"      //
            << "Iterations,"     //
            << "Time_min_us,"    //
            << "Time_max_us,"    //
            << "Time_avg_us,"    //
            << "Time_std_us,"    //
            << "AlgBW_min_GBps," //
            << "AlgBW_max_GBps," //
            << "AlgBW_avg_GBps," //
            << "AlgBW_std_GBps," //
            << "BusBW_min_GBps," //
            << "BusBW_max_GBps," //
            << "BusBW_avg_GBps," //
            << "BusBW_std_GBps";

        return oss.str();
    }

    auto text() -> std::string {
        using namespace ncclbench;

        static constexpr double SECS_TO_USECS = 1.0E6;

        std::ostringstream oss;

        oss << std::left                                                   //
            << std::setw(Result::LRG_WIDTH) << operation                   //
            << std::setw(Result::SML_WIDTH) << (blocking ? "yes" : "no")   //
            << std::setw(Result::MID_WIDTH) << data_type                   //
            << std::right                                                  //
            << std::setw(Result::LRG_WIDTH) << bytes_total                 //
            << std::setw(Result::MID_WIDTH) << elements_per_rank           //
            << std::setw(Result::MID_WIDTH) << benchmark_its               //
            << std::setw(Result::MID_WIDTH) << time.min * SECS_TO_USECS    //
            << std::setw(Result::MID_WIDTH) << time.max * SECS_TO_USECS    //
            << std::setw(Result::MID_WIDTH) << time.avg * SECS_TO_USECS    //
            << std::setw(Result::MID_WIDTH) << time.stddev * SECS_TO_USECS //
            << std::setw(Result::LRG_WIDTH) << bw_alg.min                  //
            << std::setw(Result::LRG_WIDTH) << bw_alg.max                  //
            << std::setw(Result::LRG_WIDTH) << bw_alg.avg                  //
            << std::setw(Result::LRG_WIDTH) << bw_alg.stddev               //
            << std::setw(Result::LRG_WIDTH) << bw_bus.min                  //
            << std::setw(Result::LRG_WIDTH) << bw_bus.max                  //
            << std::setw(Result::LRG_WIDTH) << bw_bus.avg                  //
            << std::setw(Result::LRG_WIDTH) << bw_bus.stddev;

        return oss.str();
    }

    auto csv() -> std::string {
        std::ostringstream oss;

        static constexpr double SECS_TO_USECS = 1.0E6;

        oss << operation << ","                   //
            << (blocking ? "yes" : "no") << ","   //
            << data_type << ","                   //
            << bytes_total << ","                 //
            << elements_per_rank << ","           //
            << benchmark_its << ","               //
            << time.min * SECS_TO_USECS << ","    //
            << time.max * SECS_TO_USECS << ","    //
            << time.avg * SECS_TO_USECS << ","    //
            << time.stddev * SECS_TO_USECS << "," //
            << bw_alg.min << ","                  //
            << bw_alg.max << ","                  //
            << bw_alg.avg << ","                  //
            << bw_alg.stddev << ","               //
            << bw_bus.min << ","                  //
            << bw_bus.max << ","                  //
            << bw_bus.avg << ","                  //
            << bw_bus.stddev;

        return oss.str();
    }
};

auto print_results_summary(const std::vector<ncclbench::Result> &results)
    -> void {}

auto main(int argc, char *argv[]) -> int {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

    CLI11_PARSE(app, argc, argv);

    ncclbench::Config config;
    config.operation = options.operation;
    config.data_type = options.data_type;
    config.blocking = options.blocking;

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

    MPI_Finalize();

    return EXIT_SUCCESS;
}
