#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>

#include "ncclbench/ncclbench.hpp"

#include "ncclbench/utils/checks.hpp"
#include "ncclbench/utils/nccl_functions.hpp"

#include "ncclbench/benchmarks/benchmarks.hpp"

namespace ncclbench {
auto Results::header() -> std::string {
    std::ostringstream oss;

    oss << std::left                               //
        << std::setw(LRG_WIDTH) << "Operation"     //
        << std::setw(SML_WIDTH) << "Blocking"      //
        << std::setw(MID_WIDTH) << "Data Type"     //
        << std::right                              //
        << std::setw(LRG_WIDTH) << "Msg Size (B)"  //
        << std::setw(MID_WIDTH) << "#Elements"     //
        << std::setw(MID_WIDTH) << "Iterations"    //
        << std::setw(MID_WIDTH) << "Min_Time (us)" //
        << std::setw(MID_WIDTH) << "Avg_Time (us)" //
        << std::setw(MID_WIDTH) << "Max_Time (us)" //
        << std::setw(MID_WIDTH) << "algBW (GB/s)"  //
        << std::setw(MID_WIDTH) << "busBW (GB/s)";

    return oss.str();
}

auto Results::csv_header() -> std::string {
    std::ostringstream oss;

    oss << "Operation,"      //
        << "Blocking,"       //
        << "Data_Type,"      //
        << "Msg_Size_Bytes," //
        << "#Elements,"      //
        << "Iterations,"     //
        << "Min_Time_us,"    //
        << "Avg_Time_us,"    //
        << "Max_Time_us,"    //
        << "algBW_GBps,"     //
        << "busBW_GBps";

    return oss.str();
}

auto Results::text() const -> std::string {
    static constexpr double SECS_TO_USECS = 1.0E6;

    std::ostringstream oss;

    oss << std::fixed << std::setprecision(PRECISION);
    oss << std::left                                         //
        << std::setw(LRG_WIDTH) << operation                 //
        << std::setw(SML_WIDTH) << (blocking ? "Yes" : "No") //
        << std::setw(MID_WIDTH) << data_type                 //
        << std::right                                        //
        << std::setw(LRG_WIDTH) << bytes_total               //
        << std::setw(MID_WIDTH) << elements_per_rank         //
        << std::setw(MID_WIDTH) << benchmark_its             //
        << std::setw(MID_WIDTH) << time_min * SECS_TO_USECS  //
        << std::setw(MID_WIDTH) << time_avg * SECS_TO_USECS  //
        << std::setw(MID_WIDTH) << time_max * SECS_TO_USECS  //
        << std::setw(MID_WIDTH) << bw_alg                    //
        << std::setw(MID_WIDTH) << bw_bus;

    return oss.str();
}

auto Results::csv() const -> std::string {
    static constexpr double SECS_TO_USECS = 1.0E6;

    std::ostringstream oss;

    oss << operation << ","                 //
        << (blocking ? "Yes" : "No") << "," //
        << data_type << ","                 //
        << bytes_total << ","               //
        << elements_per_rank << ","         //
        << benchmark_its << ","             //
        << time_min * SECS_TO_USECS << ","  //
        << time_avg * SECS_TO_USECS << ","  //
        << time_max * SECS_TO_USECS << ","  //
        << bw_alg << ","                    //
        << bw_bus;

    return oss.str();
}

static State state_;
auto state() -> State & { return state_; }

auto State::mpi_comm() -> MPI_Comm { return MPI_COMM_WORLD; }

auto State::ranks() -> int {
    if (not state_.ranks_.has_value()) {
        int ranks;
        MPICHECK(MPI_Comm_size(State::mpi_comm(), &ranks));
        state_.ranks_ = {ranks};
    }
    return state_.ranks_.value();
}

auto State::rank() -> int {
    if (not state_.rank_.has_value()) {
        int rank;
        MPICHECK(MPI_Comm_rank(State::mpi_comm(), &rank));
        state_.rank_ = {rank};
    }
    return state_.rank_.value();
}

auto State::gpu_assigned() -> int {
    if (not state_.gpu_assigned_) {
        // Calculate local rank based on hostname to select GPU
        const auto hostname = get_hostname();
        const uint64_t hostHash = get_host_hash(hostname.c_str());

        std::vector<uint64_t> hostHashes(static_cast<uint64_t>(ranks()));
        hostHashes[static_cast<size_t>(rank())] = hostHash;

        MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                               hostHashes.data(), sizeof(uint64_t), MPI_BYTE,
                               State::mpi_comm()));

        state_.gpu_assigned_ = std::count_if(
            hostHashes.begin(), hostHashes.begin() + rank(),
            [&](const uint64_t hash) { return hash == hostHash; });

        // Set the GPU device for this process
        CUDACHECK(cudaSetDevice(state_.gpu_assigned_.value()));
    }

    return state_.gpu_assigned_.value();
}

auto run(const Config &cfg) -> Results {
    // Force GPU assignment (if not already done)
    std::ignore = State::gpu_assigned();

    switch (string_to_function(cfg.operation)) {
    case NCCL_FUNCTIONS::ALL_GATHER:
        return benchmark::nccl_allgather(cfg);
    case NCCL_FUNCTIONS::ALL_REDUCE:
        return benchmark::nccl_allreduce(cfg);
    case NCCL_FUNCTIONS::ALL_TO_ALL:
        return benchmark::nccl_alltoall(cfg);
    case NCCL_FUNCTIONS::BROADCAST:
        return benchmark::nccl_broadcast(cfg);
    case NCCL_FUNCTIONS::POINT_TO_POINT:
        return benchmark::nccl_p2p(cfg);
    case NCCL_FUNCTIONS::REDUCE:
        return benchmark::nccl_reduce(cfg);
    case NCCL_FUNCTIONS::REDUCE_SCATTER:
        return benchmark::nccl_reduce_scatter(cfg);
    default:
        throw std::runtime_error{"Benchmark not implemented"};
    }
}

auto run(std::vector<Config> &cfgs) -> std::vector<Results> {
    std::vector<Results> results;
    results.reserve(cfgs.size());

    for (auto &cfg : cfgs) {
        results.push_back(run(cfg));
    }

    return results;
}

} // namespace ncclbench