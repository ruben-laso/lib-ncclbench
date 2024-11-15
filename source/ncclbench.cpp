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

    oss << "====================================================="
           "===================================="
        << '\n';
    oss << std::left                                //
        << std::setw(LRG_WIDTH) << "Operation"      //
        << std::setw(SML_WIDTH) << "Blocking"       //
        << std::setw(MID_WIDTH) << "Data_Type"      //
        << std::right                               //
        << std::setw(LRG_WIDTH) << "Msg_Size_Bytes" //
        << std::setw(MID_WIDTH) << "#Elements"      //
        << std::setw(MID_WIDTH) << "Iterations"     //
        << std::setw(MID_WIDTH) << "Min_Time (us)"  //
        << std::setw(MID_WIDTH) << "Avg_Time (us)"  //
        << std::setw(MID_WIDTH) << "Max_Time (us)"  //
        << std::setw(MID_WIDTH) << "algBW (GB/s)"   //
        << std::setw(MID_WIDTH) << "busBW (GB/s)"   //
        << '\n';

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
        << "Min_Time (us),"  //
        << "Avg_Time (us),"  //
        << "Max_Time (us),"  //
        << "algBW (GB/s),"   //
        << "busBW (GB/s)"    //
        << '\n';

    return oss.str();
}

auto Results::text() const -> std::string {
    std::ostringstream oss;

    oss << std::fixed << std::setprecision(PRECISION);
    oss << std::left                                         //
        << std::setw(LRG_WIDTH) << operation                 //
        << std::setw(SML_WIDTH) << (blocking ? "Yes" : "No") //
        << std::setw(MID_WIDTH) << data_type                 //
        << std::right                                        //
        << std::setw(LRG_WIDTH) << bytes_total               //
        << std::setw(MID_WIDTH) << elements_per_rank         //
        << std::setw(MID_WIDTH) << iterations                //
        << std::setw(MID_WIDTH) << time_min                  //
        << std::setw(MID_WIDTH) << time_avg                  //
        << std::setw(MID_WIDTH) << time_max                  //
        << std::setw(MID_WIDTH) << bw_alg                    //
        << std::setw(MID_WIDTH) << bw_bus                    //
        << '\n';

    return oss.str();
}

auto Results::csv() const -> std::string {
    std::ostringstream oss;

    oss << operation << ","                 //
        << (blocking ? "Yes" : "No") << "," //
        << data_type << ","                 //
        << bytes_total << ","               //
        << elements_per_rank << ","         //
        << iterations << ","                //
        << time_min << ","                  //
        << time_avg << ","                  //
        << time_max << ","                  //
        << bw_alg << ","                    //
        << bw_bus << '\n';

    return oss.str();
}

static State state_;
auto state() -> State & { return state_; }

auto State::mpi_comm() -> MPI_Comm { return MPI_COMM_WORLD; }

auto State::nccl_comm() -> ncclComm_t {
    if (not state_.nccl_comm_.has_value()) {
        const auto nccl_id = State::nccl_id();
        ncclComm_t comm;
        NCCLCHECK(
            ncclCommInitRank(&comm, State::ranks(), nccl_id, State::rank()));
        state_.nccl_comm_ = {comm};
    }
    return state_.nccl_comm_.value();
}

auto State::nccl_id() -> ncclUniqueId {
    if (not state_.nccl_id_.has_value()) {
        ncclUniqueId id;
        NCCLCHECK(ncclGetUniqueId(&id));
        state_.nccl_id_ = {id};
    }
    return state_.nccl_id_.value();
}

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
        char hostname[1024];
        get_hostname(hostname, 1024);
        const uint64_t hostHash = get_host_hash(hostname);

        uint64_t hostHashes[ranks()];
        hostHashes[rank()] = hostHash;

        MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashes,
                               sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));

        state_.gpu_assigned_ = std::count_if(
            hostHashes, hostHashes + rank(),
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
        return benchmarks::nccl_allgather(cfg);
    case NCCL_FUNCTIONS::ALL_REDUCE:
        return benchmarks::nccl_allreduce(cfg);
    case NCCL_FUNCTIONS::ALL_TO_ALL:
        return benchmarks::nccl_alltoall(cfg);
    case NCCL_FUNCTIONS::BROADCAST:
        return benchmarks::nccl_broadcast(cfg);
    case NCCL_FUNCTIONS::POINT_TO_POINT:
        return benchmarks::nccl_p2p(cfg);
    case NCCL_FUNCTIONS::REDUCE:
        return benchmarks::nccl_reduce(cfg);
    case NCCL_FUNCTIONS::REDUCE_SCATTER:
        return benchmarks::nccl_reduce_scatter(cfg);
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