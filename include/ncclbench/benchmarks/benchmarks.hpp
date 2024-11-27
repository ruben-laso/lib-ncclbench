#pragma once

#include "ncclbench/ncclbench.hpp"

namespace ncclbench::benchmark {
auto nccl_allgather(const Config &cfg) -> Results;
auto nccl_allreduce(const Config &cfg) -> Results;
auto nccl_alltoall(const Config &cfg) -> Results;
auto nccl_broadcast(const Config &cfg) -> Results;
auto nccl_p2p(const Config &cfg) -> Results;
auto nccl_reduce(const Config &cfg) -> Results;
auto nccl_reduce_scatter(const Config &cfg) -> Results;
} // namespace ncclbench::benchmark
