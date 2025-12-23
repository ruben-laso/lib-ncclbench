#pragma once

#include <ncclbench/ncclbench.hpp>

#include <yaml-cpp/yaml.h>

#include "Options.hpp"

namespace cfgs {
namespace yaml {
inline auto load_operation_cfg(const YAML::Node &node, const Options &options) {
    std::vector<ncclbench::Config> cfgs;

    const auto sizes = [&]() {
        // First try as vector
        // Second, as a single size
        // Else, throw error
        try {
            return node["sizes"].as<std::vector<size_t>>();
        } catch (const YAML::Exception &e) {
            return std::vector<size_t>{node["sizes"].as<size_t>()};
        }
    }();

    for (const auto &size : sizes) {
        ncclbench::Config cfg;
        cfg.bytes_total = size;

        cfg.operation = node["operation"].as<std::string>();
        cfg.data_type = node["data-type"].as<std::string>();

        // Optional parameters
        cfg.blocking =
            node["blocking"] ? node["blocking"].as<bool>() : options.blocking;
        cfg.group = node["group"] ? node["group"].as<bool>() : options.group;
        cfg.warmup_its =
            node["warmup-its"]
                ? std::optional<size_t>{node["warmup-its"].as<size_t>()}
                : std::nullopt;
        cfg.benchmark_its =
            node["benchmark-its"]
                ? std::optional<size_t>{node["benchmark-its"].as<size_t>()}
                : std::nullopt;
        cfg.warmup_secs =
            node["warmup-secs"]
                ? std::optional<double>{node["warmup-secs"].as<double>()}
                : std::nullopt;
        cfg.benchmark_secs =
            node["benchmark-secs"]
                ? std::optional<double>{node["benchmark-secs"].as<double>()}
                : std::nullopt;

        cfgs.push_back(cfg);
    }

    return cfgs;
}

inline auto generate_cfgs(const Options &options) {
    // Load configurations from YAML file
    // TODO: implement this function
    YAML::Node config = YAML::LoadFile(options.config_file.string());

    std::vector<ncclbench::Config> cfgs;

    for (const auto &node : config) {
        const auto single_cfgs = load_operation_cfg(node, options);
        cfgs.insert(cfgs.end(), single_cfgs.begin(), single_cfgs.end());
    }

    return cfgs;
}
} // namespace yaml

namespace args {
inline auto generate_cfgs(const Options &options) {
    const auto num_cfgs = options.sizes_bytes.size();

    std::vector<ncclbench::Config> cfgs(num_cfgs);

    auto comm = options.reuse_comm
                    ? std::optional<ncclComm_t>{ncclbench::State::nccl_comm()}
                    : std::optional<ncclComm_t>{std::nullopt};

    for (size_t i = 0; i < num_cfgs; i++) {
        auto &cfg = cfgs[i];
        cfg.operation = options.operation;
        cfg.data_type = options.data_type;
        cfg.bytes_total = options.sizes_bytes[i];
        cfg.blocking = options.blocking;
        cfg.group = options.group;
        cfg.comm = comm;
        // Handle warmup iterations or time
        if (not options.warmup_its.empty() and options.warmup_its[i] > 0) {
            cfg.warmup_its = options.warmup_its[i];
        }
        if (not options.warmup_secs.empty() and options.warmup_secs[i] > 0.0) {
            cfg.warmup_secs = options.warmup_secs[i];
        }

        // Handle benchmark iterations or time
        if (not options.benchmark_its.empty() and
            options.benchmark_its[i] > 0) {
            cfg.benchmark_its = options.benchmark_its[i];
        }
        if (not options.benchmark_secs.empty() and
            options.benchmark_secs[i] > 0.0) {
            cfg.benchmark_secs = options.benchmark_secs[i];
        }
    }

    return cfgs;
}
} // namespace args

inline auto generate_cfgs(const Options &options) {
    if (not options.config_file.empty()) {
        return yaml::generate_cfgs(options);
    } else {
        return args::generate_cfgs(options);
    }
}
} // namespace cfgs