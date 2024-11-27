#include "ncclbench/benchmarks/common.hpp"

namespace ncclbench::benchmark {

auto nccl_reduce_scatter(const Config &cfg) -> Results {
    static constexpr int ALIGN = 4;

    const auto bytes_per_rank =
        cfg.bytes_total / (State::ranks() * ALIGN) * ALIGN;
    const auto elements_per_rank = types::bytes_to_elements(
        bytes_per_rank, types::str_to_mpi(cfg.data_type));

    Sizes sizes{};

    sizes.bytes_total = cfg.bytes_total;
    sizes.bytes_per_rank = bytes_per_rank;
    sizes.elements_per_rank = elements_per_rank;
    sizes.bytes_send = bytes_per_rank * State::ranks();
    sizes.elements_send = elements_per_rank * State::ranks();
    sizes.bytes_recv = bytes_per_rank;
    sizes.elements_recv = elements_per_rank;

    const auto nccl_call = [&](const void *sendbuff, void *recvbuff,
                               size_t count, ncclDataType_t datatype,
                               ncclComm_t comm, cudaStream_t stream) {
        static constexpr ncclRedOp_t op = ncclSum;
        NCCLCHECK(ncclReduceScatter(sendbuff, recvbuff, count, datatype, op,
                                    comm, stream));
    };

    const auto bw_factor = []() {
        return (State::ranks() - 1.0) / State::ranks();
    };

    return run_benchmark(cfg, sizes, nccl_call, bw_factor);
}

} // namespace ncclbench::benchmark
