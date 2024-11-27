#include "ncclbench/benchmarks/common.hpp"

namespace ncclbench::benchmark {

auto nccl_broadcast(const Config &cfg) -> Results {
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
        static constexpr auto root = 0;
        NCCLCHECK(ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm,
                                stream));
    };

    const auto bw_factor = []() { return 1.0; };

    return run_benchmark(cfg, sizes, nccl_call, bw_factor);
}

} // namespace ncclbench::benchmark
