#pragma once

#include <map>
#include <stdexcept>
#include <string_view>

#include <mpi.h>

#include "ncclbench/ncclbench_export.hpp"
#include "ncclbench/xccl.hpp"

#if defined(__CUDA_BF16_TYPES_EXIST__) &&                                      \
    NCCL_VERSION_CODE >= NCCL_VERSION(2, 10, 0)
#ifndef NCCL_BF16_TYPES_EXIST
#define NCCL_BF16_TYPES_EXIST
#endif
#endif

namespace ncclbench::types {

static constexpr std::string_view NCCL_INT_8 = "int8";
static constexpr std::string_view NCCL_UINT_8 = "uint8";
static constexpr std::string_view NCCL_INT_32 = "int32";
static constexpr std::string_view NCCL_UINT_32 = "uint32";
static constexpr std::string_view NCCL_INT_64 = "int64";
static constexpr std::string_view NCCL_UINT_64 = "uint64";
static constexpr std::string_view NCCL_HALF = "half";
static constexpr std::string_view NCCL_FLOAT = "float";
static constexpr std::string_view NCCL_DOUBLE = "double";
#if defined(NCCL_BF16_TYPES_EXIST)
static constexpr std::string_view NCCL_BFLOAT16 = "bfloat16";
#endif

enum NCCL_TYPES {
    INT8,
    UINT8,
    INT32,
    UINT32,
    INT64,
    UINT64,
    HALF,
    FLOAT,
    DOUBLE,
#if defined(NCCL_BF16_TYPES_EXIST)
    BFLOAT16,
#endif
    MAX_TYPES
};

static constexpr size_t NUM_NCCL_TYPES =
    static_cast<size_t>(NCCL_TYPES::MAX_TYPES);

inline ncclDataType_t str_to_nccl(const std::string_view &str) {
    if (str == NCCL_INT_8) {
        return ncclInt8;
    }
    if (str == NCCL_UINT_8) {
        return ncclUint8;
    }
    if (str == NCCL_INT_32) {
        return ncclInt32;
    }
    if (str == NCCL_UINT_32) {
        return ncclUint32;
    }
    if (str == NCCL_INT_64) {
        return ncclInt64;
    }
    if (str == NCCL_UINT_64) {
        return ncclUint64;
    }
    if (str == NCCL_HALF) {
        return ncclHalf;
    }
    if (str == NCCL_FLOAT) {
        return ncclFloat;
    }
    if (str == NCCL_DOUBLE) {
        return ncclDouble;
    }
#if defined(NCCL_BF16_TYPES_EXIST)
    if (str == NCCL_BFLOAT16) {
        return ncclBfloat16;
    }
#endif

    throw std::runtime_error("Unsupported datatype");
}

inline auto size_of(const MPI_Datatype datatype) -> size_t {
    int size;
    MPI_Type_size(datatype, &size);
    return static_cast<size_t>(size);
}

inline auto size_of(const ncclDataType_t datatype) -> size_t {
    switch (datatype) {
    case ncclChar:
#if NCCL_MAJOR >= 2
    // case ncclInt8:
    case ncclUint8:
#endif
        return 1;
    case ncclHalf:
#if defined(__CUDA_BF16_TYPES_EXIST__)
    case ncclBfloat16:
#endif
        // case ncclFloat16:
        return 2;
    case ncclInt:
    case ncclFloat:
#if NCCL_MAJOR >= 2
    // case ncclInt32:
    case ncclUint32:
        // case ncclFloat32:
#endif
        return 4;
    case ncclInt64:
    case ncclUint64:
    case ncclDouble:
        // case ncclFloat64:
        return 8;
    default:
        return 0;
    }
}

inline auto bytes_to_elements(const size_t bytes, const ncclDataType_t datatype)
    -> size_t {
    return bytes / size_of(datatype);
}

} // namespace ncclbench::types
