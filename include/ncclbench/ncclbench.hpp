#pragma once

#include <array>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <mpi.h>

#include "ncclbench/ncclbench_export.hpp"

#include "ncclbench/utils/nccl_functions.hpp"
#include "ncclbench/utils/types.hpp"

#include "ncclbench/xccl.hpp"

namespace ncclbench {

NCCLBENCH_EXPORT
static constexpr std::array<const char *const, functions::NUM_NCCL_FUNCTIONS>
    SUPPORTED_OPERATIONS = {
        functions::NCCL_ALL_GATHER,     //
        functions::NCCL_ALL_REDUCE,     //
        functions::NCCL_ALL_TO_ALL,     //
        functions::NCCL_BROADCAST,      //
        functions::NCCL_POINT_TO_POINT, //
        functions::NCCL_REDUCE_SCATTER, //
        functions::NCCL_REDUCE,         //
};

NCCLBENCH_EXPORT
static constexpr std::array<const char *const, types::NUM_NCCL_TYPES>
    SUPPORTED_DATA_TYPES = {
        types::NCCL_INT_8,   //
        types::NCCL_UINT_8,  //
        types::NCCL_INT_32,  //
        types::NCCL_UINT_32, //
        types::NCCL_INT_64,  //
        types::NCCL_UINT_64, //
        types::NCCL_HALF,    //
        types::NCCL_FLOAT,   //
        types::NCCL_DOUBLE,  //
#if defined(NCCL_BF16_TYPES_EXIST)
        types::NCCL_BFLOAT16, //
#endif
};

struct NCCLBENCH_EXPORT Config {
    std::string operation;
    bool blocking;
    std::string data_type;
    size_t bytes_total;
    std::optional<size_t> warmup_its;
    std::optional<size_t> benchmark_its;
    std::optional<double> warmup_secs;
    std::optional<double> benchmark_secs;
};

struct Sizes {
    size_t bytes_total;

    size_t bytes_per_rank;
    size_t elements_per_rank;

    size_t bytes_send;
    size_t elements_send;

    size_t bytes_recv;
    size_t elements_recv;
};

struct NCCLBENCH_EXPORT Result {
    static constexpr size_t SML_WIDTH = 10;
    static constexpr size_t MID_WIDTH = 15;
    static constexpr size_t LRG_WIDTH = 20;

    static constexpr size_t PRECISION = 4;

    std::string operation;
    bool blocking;
    std::string data_type;
    size_t bytes_total;
    size_t elements_per_rank;
    size_t benchmark_its;
    double time;
    double bw_alg;
    double bw_bus;

    [[nodiscard]] static auto header() -> std::string;
    [[nodiscard]] static auto csv_header() -> std::string;

    [[nodiscard]] auto text() const -> std::string;
    [[nodiscard]] auto csv() const -> std::string;
};

class State {
    std::optional<int> ranks_ = std::nullopt;
    std::optional<int> rank_ = std::nullopt;

    std::optional<int> gpu_assigned_ = std::nullopt;

  public:
    [[nodiscard]] static auto mpi_comm() -> MPI_Comm;
    [[nodiscard]] static auto ranks() -> int;
    [[nodiscard]] static auto rank() -> int;
    [[nodiscard]] static auto gpu_assigned() -> int;
};

NCCLBENCH_EXPORT auto run(const Config &cfg) -> std::vector<Result>;
NCCLBENCH_EXPORT auto run(std::vector<Config> &cfgs)
    -> std::vector<std::vector<Result>>;

NCCLBENCH_EXPORT auto state() -> State &;

} // namespace ncclbench