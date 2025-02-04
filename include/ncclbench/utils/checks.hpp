#pragma once

#include <cassert>

#include <cuda.h>
#include <mpi.h>

#include <tuple>
#include <utility>

#include "../xccl.hpp"
#include "host.hpp"

#define MPICHECK(stmt)                                                         \
    {                                                                          \
        int mpi_errno = (stmt);                                                \
        if (MPI_SUCCESS != mpi_errno) {                                        \
            std::ignore =                                                      \
                fprintf(stderr, "[%s:%d] MPI call failed with %d \n",          \
                        __FILE__, __LINE__, mpi_errno);                        \
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                           \
        }                                                                      \
    }

#define CUDACHECK(cmd)                                                         \
    {                                                                          \
        cudaError_t err = cmd;                                                 \
        if (err != cudaSuccess) {                                              \
            const auto check_hostname = get_hostname();                        \
            std::ignore =                                                      \
                fprintf(stderr, "%s: Test CUDA failure %s:%d '%s'\n",          \
                        check_hostname.c_str(), __FILE__, __LINE__,            \
                        cudaGetErrorString(err));                              \
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                           \
        }                                                                      \
    }

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 13, 0)
#define NCCLCHECK(cmd)                                                         \
    {                                                                          \
        ncclResult_t res = cmd;                                                \
        if (res != ncclSuccess) {                                              \
            const auto check_hostname = get_hostname();                        \
            std::ignore =                                                      \
                fprintf(stderr,                                                \
                        "%s: Test NCCL failure %s:%d "                         \
                        "'%s / %s'\n",                                         \
                        check_hostname.c_str(), __FILE__, __LINE__,            \
                        ncclGetErrorString(res), ncclGetLastError(NULL));      \
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                           \
        }                                                                      \
    }
#else
#define NCCLCHECK(cmd)                                                         \
    do {                                                                       \
        ncclResult_t res = cmd;                                                \
        if (res != ncclSuccess) {                                              \
            char hostname[1024];                                               \
            getHostName(hostname, 1024);                                       \
            printf("%s: Test NCCL failure %s:%d '%s'\n", hostname, __FILE__,   \
                   __LINE__, ncclGetErrorString(res));                         \
        }                                                                      \
    } while (0)
#endif