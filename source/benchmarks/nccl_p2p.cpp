#include "ncclbench/benchmarks/common.hpp"

namespace ncclbench::benchmark {

auto nccl_p2p(const Config &cfg) -> Results {
    const auto rank = State::rank();
    const auto ranks = State::ranks();

    if (ranks % 2 != 0) {
        throw std::runtime_error("This benchmark requires an even number of "
                                 "processes.");
    }

    const auto bytes_per_rank = cfg.bytes_total / ranks;
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
        const int peer = (rank + 1) % 2;
        if (rank == 0) {
            NCCLCHECK(ncclSend(sendbuff, count, datatype, peer, comm, stream));
            NCCLCHECK(ncclRecv(recvbuff, count, datatype, peer, comm, stream));
        } else if (rank == 1) {
            NCCLCHECK(ncclRecv(recvbuff, count, datatype, peer, comm, stream));
            NCCLCHECK(ncclSend(sendbuff, count, datatype, peer, comm, stream));
        }
    };

    const auto bw_factor = []() { return 1.0; };

    return run_benchmark(cfg, sizes, nccl_call, bw_factor);
}

} // namespace ncclbench::benchmark
