#pragma once

#include <cassert>

#include <cuda.h>
#include <mpi.h>

#include "../xccl.hpp"
#include "host.hpp"

#define MPICHECK(stmt)                                                         \
    do {                                                                       \
        int mpi_errno = (stmt);                                                \
        if (MPI_SUCCESS != mpi_errno) {                                        \
            fprintf(stderr, "[%s:%d] MPI call failed with %d \n", __FILE__,    \
                    __LINE__, mpi_errno);                                      \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
        assert(MPI_SUCCESS == mpi_errno);                                      \
    } while (0)

#define CUDACHECK(cmd)                                                         \
    do {                                                                       \
        cudaError_t err = cmd;                                                 \
        if (err != cudaSuccess) {                                              \
            char hostname[1024];                                               \
            get_hostname(hostname, 1024);                                      \
            printf("%s: Test CUDA failure %s:%d '%s'\n", hostname, __FILE__,   \
                   __LINE__, cudaGetErrorString(err));                         \
        }                                                                      \
    } while (0)

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 13, 0)
#define NCCLCHECK(cmd)                                                         \
    do {                                                                       \
        ncclResult_t res = cmd;                                                \
        if (res != ncclSuccess) {                                              \
            char hostname[1024];                                               \
            get_hostname(hostname, 1024);                                      \
            printf("%s: Test NCCL failure %s:%d "                              \
                   "'%s / %s'\n",                                              \
                   hostname, __FILE__, __LINE__, ncclGetErrorString(res),      \
                   ncclGetLastError(NULL));                                    \
        }                                                                      \
    } while (0)
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