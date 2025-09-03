#pragma once

#include <cstdint>
#include <stdexcept>
#include <string_view>

#include "ncclbench/ncclbench_export.hpp"

namespace ncclbench::functions {

static constexpr std::string_view NCCL_ALL_GATHER = "ncclAllGather";
static constexpr std::string_view NCCL_ALL_REDUCE = "ncclAllReduce";
static constexpr std::string_view NCCL_ALL_TO_ALL = "ncclAllToAll";
static constexpr std::string_view NCCL_BROADCAST = "ncclBroadcast";
static constexpr std::string_view NCCL_POINT_TO_POINT = "ncclPointToPoint";
static constexpr std::string_view NCCL_REDUCE_SCATTER = "ncclReduceScatter";
static constexpr std::string_view NCCL_REDUCE = "ncclReduce";

enum NCCL_FUNCTIONS {
    ALL_GATHER,
    ALL_REDUCE,
    ALL_TO_ALL,
    BROADCAST,
    POINT_TO_POINT,
    REDUCE_SCATTER,
    REDUCE,
    NCCL_FUNCTIONS_MAX
};

static constexpr size_t NUM_NCCL_FUNCTIONS =
    static_cast<size_t>(NCCL_FUNCTIONS::NCCL_FUNCTIONS_MAX);

inline auto string_to_function(std::string const &function) -> NCCL_FUNCTIONS {
    if (function == NCCL_ALL_GATHER) {
        return NCCL_FUNCTIONS::ALL_GATHER;
    }
    if (function == NCCL_ALL_REDUCE) {
        return NCCL_FUNCTIONS::ALL_REDUCE;
    }
    if (function == NCCL_ALL_TO_ALL) {
        return NCCL_FUNCTIONS::ALL_TO_ALL;
    }
    if (function == NCCL_BROADCAST) {
        return NCCL_FUNCTIONS::BROADCAST;
    }
    if (function == NCCL_POINT_TO_POINT) {
        return NCCL_FUNCTIONS::POINT_TO_POINT;
    }
    if (function == NCCL_REDUCE_SCATTER) {
        return NCCL_FUNCTIONS::REDUCE_SCATTER;
    }
    if (function == NCCL_REDUCE) {
        return NCCL_FUNCTIONS::REDUCE;
    }
    throw std::runtime_error{"Unknown NCCL function"};
}

inline auto function_to_string(NCCL_FUNCTIONS const function)
    -> std::string_view {
    switch (function) {
    case NCCL_FUNCTIONS::ALL_GATHER:
        return NCCL_ALL_GATHER;
    case NCCL_FUNCTIONS::ALL_REDUCE:
        return NCCL_ALL_REDUCE;
    case NCCL_FUNCTIONS::ALL_TO_ALL:
        return NCCL_ALL_TO_ALL;
    case NCCL_FUNCTIONS::BROADCAST:
        return NCCL_BROADCAST;
    case NCCL_FUNCTIONS::POINT_TO_POINT:
        return NCCL_POINT_TO_POINT;
    case NCCL_FUNCTIONS::REDUCE_SCATTER:
        return NCCL_REDUCE_SCATTER;
    case NCCL_FUNCTIONS::REDUCE:
        return NCCL_REDUCE;
    }
    throw std::runtime_error{"Unknown NCCL function"};
}
} // namespace ncclbench::functions
