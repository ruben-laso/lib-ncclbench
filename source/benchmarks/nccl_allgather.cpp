#include "ncclbench/benchmarks/common.hpp"

namespace ncclbench::benchmark {

auto nccl_allgather(const Config &cfg) -> Results {
    static constexpr size_t ALIGN = 4;

    const auto uranks = static_cast<size_t>(State::ranks());

    const auto bytes_per_rank = cfg.bytes_total / (uranks * ALIGN) * ALIGN;
    const auto elements_per_rank = types::bytes_to_elements(
        bytes_per_rank, types::str_to_mpi(cfg.data_type));

    Sizes sizes{};

    sizes.bytes_total = cfg.bytes_total;
    sizes.bytes_per_rank = bytes_per_rank;
    sizes.elements_per_rank = elements_per_rank;
    sizes.bytes_send = bytes_per_rank;
    sizes.elements_send = elements_per_rank;
    sizes.bytes_recv = bytes_per_rank * uranks;
    sizes.elements_recv = elements_per_rank * uranks;

    const auto nccl_call = [&](const void *sendbuff, void *recvbuff,
                               size_t count, ncclDataType_t datatype,
                               ncclComm_t comm, cudaStream_t stream) {
        NCCLCHECK(
            ncclAllGather(sendbuff, recvbuff, count, datatype, comm, stream));
    };

    const auto bw_factor = []() {
        const auto dranks = static_cast<double>(State::ranks());
        return (dranks - 1.0) / dranks;
    };

    return run_benchmark(cfg, sizes, nccl_call, bw_factor);
}

} // namespace ncclbench::benchmark
