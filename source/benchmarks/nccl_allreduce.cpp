#include "ncclbench/benchmarks/common.hpp"

namespace ncclbench::benchmark {

auto nccl_allreduce(const Config &cfg) -> std::vector<Result> {
    const auto bytes_per_rank = cfg.bytes_total;
    const auto elements_per_rank = types::bytes_to_elements(
        bytes_per_rank, types::str_to_mpi(cfg.data_type));

    Sizes sizes{};

    sizes.bytes_total = cfg.bytes_total;
    sizes.bytes_per_rank = bytes_per_rank;
    sizes.elements_per_rank = elements_per_rank;
    sizes.bytes_send = bytes_per_rank;
    sizes.elements_send = elements_per_rank;
    sizes.bytes_recv = bytes_per_rank;
    sizes.elements_recv = elements_per_rank;

    const auto nccl_call = [&](const void *sendbuff, void *recvbuff,
                               size_t count, ncclDataType_t datatype,
                               ncclComm_t comm, cudaStream_t stream) {
        static constexpr ncclRedOp_t op = ncclSum;
        NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm,
                                stream));
    };

    const auto bw_factor = []() {
        const auto dranks = static_cast<double>(State::ranks());
        return (2.0 * (dranks - 1.0)) / dranks;
    };

    return run_benchmark(cfg, sizes, nccl_call, bw_factor);
}

} // namespace ncclbench::benchmark