#include "ncclbench/benchmarks/common.hpp"

namespace ncclbench::benchmark {

void nccl_allgather(const Config &cfg) {
    static constexpr int ALIGN = 4;

    const auto bytes_per_rank =
        cfg.bytes_total / (State::ranks() * ALIGN) * ALIGN;
    const auto elements_per_rank = types::bytes_to_elements(
        bytes_per_rank, types::str_to_mpi(cfg.data_type));

    Sizes sizes = {
        .bytes_total = cfg.bytes_total,
        .bytes_per_rank = bytes_per_rank,
        .elements_per_rank = elements_per_rank,
        .bytes_send = bytes_per_rank,
        .elements_send = elements_per_rank,
        .bytes_recv = bytes_per_rank * State::ranks(),
        .elements_recv = elements_per_rank * State::ranks(),
    };

    const auto nccl_call = [&](const void *sendbuff, void *recvbuff, size_t count,
                               ncclDataType_t datatype, ncclComm_t comm,
                               cudaStream_t stream) {
        NCCLCHECK(
            ncclAllGather(sendbuff, recvbuff, count, datatype, comm, stream));
    };

    const auto bw_factor = []() {
        return (State::ranks() - 1.0) / State::ranks();
    };

    run_benchmark(cfg, sizes, nccl_call, bw_factor);
}

} // namespace ncclbench::benchmark
