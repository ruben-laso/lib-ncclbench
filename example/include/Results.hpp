#pragma once

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <ncclbench/ncclbench.hpp>

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
        blocking = results[0].blocking;
        data_type = results[0].data_type;
        bytes_total = results[0].bytes_total;
        elements_per_rank = results[0].elements_per_rank;
        benchmark_its = std::transform_reduce(
            results.begin(), results.end(), 0, std::plus<>{},
            [](const auto &a) { return a.benchmark_its; });

        // Time stats
        time.min = std::min_element(results.begin(), results.end(),
                                    [](const auto &a, const auto &b) {
                                        return a.time < b.time;
                                    })
                       ->time;
        time.max = std::max_element(results.begin(), results.end(),
                                    [](const auto &a, const auto &b) {
                                        return a.time < b.time;
                                    })
                       ->time;
        time.avg = std::transform_reduce(results.begin(), results.end(), 0.0,
                                         std::plus<>{},
                                         [](const auto &a) { return a.time; }) /
                   results.size();
        time.stddev =
            std::sqrt(std::transform_reduce(
                          results.begin(), results.end(), 0.0, std::plus<>{},
                          [avg = time.avg](const auto &a) {
                              return std::pow(a.time - avg, 2);
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
            << std::setw(Result::SML_WIDTH) << (blocking ? "Yes" : "No")   //
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
            << (blocking ? "Yes" : "No") << ","   //
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
