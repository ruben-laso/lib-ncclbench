#pragma once

#include "ncclbench/ncclbench.hpp"

namespace ncclbench::benchmark {
auto nccl_allgather(const Config &cfg) -> std::vector<Result>;
auto nccl_allreduce(const Config &cfg) -> std::vector<Result>;
auto nccl_alltoall(const Config &cfg) -> std::vector<Result>;
auto nccl_broadcast(const Config &cfg) -> std::vector<Result>;
auto nccl_p2p(const Config &cfg) -> std::vector<Result>;
auto nccl_reduce(const Config &cfg) -> std::vector<Result>;
auto nccl_reduce_scatter(const Config &cfg) -> std::vector<Result>;
} // namespace ncclbench::benchmark
