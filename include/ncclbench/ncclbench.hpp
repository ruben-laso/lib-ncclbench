#pragma once

#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <mpi.h>

#include "ncclbench/ncclbench_export.hpp"

#if defined(USE_RCCL)
#include <rccl/rccl.h>
#else // USE_NCCL by default
#include <nccl.h>
#endif

namespace ncclbench {

struct NCCLBENCH_EXPORT Config {
    std::string operation;
    bool blocking;
    std::string data_type;
    size_t bytes_total;
    std::variant<size_t, double> warmup_its_or_secs;
    std::variant<size_t, double> benchmark_its_or_secs;
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

struct NCCLBENCH_EXPORT Results {
  private:
    static constexpr size_t SML_WIDTH = 10;
    static constexpr size_t MID_WIDTH = 15;
    static constexpr size_t LRG_WIDTH = 20;

    static constexpr size_t PRECISION = 2;

  public:
    std::string operation;
    bool blocking;
    std::string data_type;
    size_t bytes_total;
    size_t elements_per_rank;
    size_t warmup_its;
    size_t benchmark_its;
    double time_min;
    double time_max;
    double time_avg;
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

NCCLBENCH_EXPORT auto run(const Config &cfg) -> Results;
NCCLBENCH_EXPORT auto run(std::vector<Config> &cfgs) -> std::vector<Results>;

NCCLBENCH_EXPORT auto state() -> State &;

} // namespace ncclbench