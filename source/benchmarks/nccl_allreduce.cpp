#include "ncclbench/benchmarks/common.hpp"

namespace ncclbench::benchmark {

void nccl_allreduce(const Config &cfg) {
    const auto bytes_per_rank = cfg.bytes_total;
    const auto elements_per_rank = types::bytes_to_elements(
        bytes_per_rank, types::str_to_mpi(cfg.data_type));

    Sizes sizes = {
        .bytes_total = cfg.bytes_total,
        .bytes_per_rank = bytes_per_rank,
        .elements_per_rank = elements_per_rank,
        .bytes_send = bytes_per_rank,
        .elements_send = elements_per_rank,
        .bytes_recv = bytes_per_rank,
        .elements_recv = elements_per_rank,
    };

    const auto nccl_call = [&](const void *sendbuff, void *recvbuff,
                               size_t count, ncclDataType_t datatype,
                               ncclComm_t comm, cudaStream_t stream) {
        static constexpr ncclRedOp_t op = ncclSum;
        NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm,
                                stream));
    };

    const auto bw_factor = []() {
        return (2.0 * (State::ranks() - 1.0)) / State::ranks();
    };

    run_benchmark(cfg, sizes, nccl_call, bw_factor);
}

} // namespace ncclbench::benchmark