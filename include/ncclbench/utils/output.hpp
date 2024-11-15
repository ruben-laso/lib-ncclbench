#pragma once

#include <iostream>

template <typename C, typename OStream = std::ostream>
void print_container(const C &container, OStream &out = std::cout) {
    out << "[ ";
    for (const auto &elem : container) {
        out << elem << " ";
    }
    out << "]";
}

template <typename C, typename OStream = std::ostream>
void print_list(const C &container, OStream &out = std::cout) {
    auto it = container.begin();
    if (it != container.end()) {
        out << *it;
        ++it;
    }
    for (; it != container.end(); ++it) {
        out << ", " << *it;
    }
}

void print_ordered(const std::string &mess, std::ostream &ostream = std::cout);

//void printGPUinfo(const CmdOptions &opts);
//
//void printMetaInfo(const CmdOptions &opts, const bool csv);
//
//void printResultsHeader(const CmdOptions &opts, const bool csv);
//
//void printResults(const CmdOptions &opts, const benchArgs &args,
//                  const PerformanceMetrics &metrics, const bool csv);