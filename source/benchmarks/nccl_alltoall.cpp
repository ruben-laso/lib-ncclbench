#include "ncclbench/benchmarks/common.hpp"

namespace ncclbench::benchmark {

auto nccl_alltoall(const Config &cfg) -> std::vector<Result> {
    const auto ranks = State::ranks();
    const auto uranks = static_cast<size_t>(ranks);

    const auto bytes_per_rank = cfg.bytes_total / uranks;
    const auto elements_per_rank = types::bytes_to_elements(
        bytes_per_rank, types::str_to_nccl(cfg.data_type));

    Sizes sizes{};

    sizes.bytes_total = cfg.bytes_total;
    sizes.bytes_per_rank = bytes_per_rank;
    sizes.elements_per_rank = elements_per_rank;
    sizes.bytes_send = bytes_per_rank * uranks;
    sizes.elements_send = elements_per_rank * uranks;
    sizes.bytes_recv = bytes_per_rank * uranks;
    sizes.elements_recv = elements_per_rank * uranks;

    const auto nccl_call = [&](const void *sendbuff, void *recvbuff,
                               size_t count, ncclDataType_t datatype,
                               ncclComm_t comm, cudaStream_t stream) {
        const auto rankOffset =
            count * types::size_of(types::str_to_nccl(cfg.data_type));

#if NCCL_MAJOR >= 2 && NCCL_MINOR >= 7
        NCCLCHECK(ncclGroupStart());
#endif
        for (size_t r = 0; r < uranks; r++) {
            const char *sendbuff_r =
                static_cast<const char *>(sendbuff) + r * rankOffset;
            char *recvbuff_r = static_cast<char *>(recvbuff) + r * rankOffset;
            NCCLCHECK(ncclSend(sendbuff_r, count, datatype, static_cast<int>(r),
                               comm, stream));
            NCCLCHECK(ncclRecv(recvbuff_r, count, datatype, static_cast<int>(r),
                               comm, stream));
        }
#if NCCL_MAJOR >= 2 && NCCL_MINOR >= 7
        NCCLCHECK(ncclGroupEnd());
#endif
    };

    const auto bw_factor = []() {
        const auto dranks = static_cast<double>(State::ranks());
        return (dranks - 1.0) / dranks;
    };

    return run_benchmark(cfg, sizes, nccl_call, bw_factor);
}

} // namespace ncclbench::benchmark
