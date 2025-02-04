#pragma once

#include <numeric>
#include <stdexcept>

#include <mpi.h>

namespace ncclbench::utils {

template <typename C> void init_data(C &container) {
    using T = typename C::value_type;

    std::iota(container.begin(), container.end(), T{});
}

template <typename T> void iota(T *ptr, size_t count) {
    std::iota(ptr, ptr + count, T{});
}

// static void init_data(void *ptr, MPI_Datatype datatype, size_t count) {
//     if (datatype == MPI_BYTE) {
//         iota(static_cast<char *>(ptr), count);
//     } else if (datatype == MPI_CHAR) {
//         iota(static_cast<char *>(ptr), count);
//     } else if (datatype == MPI_INT) {
//         iota(static_cast<int *>(ptr), count);
//     } else if (datatype == MPI_FLOAT) {
//         iota(static_cast<float *>(ptr), count);
//     } else if (datatype == MPI_DOUBLE) {
//         iota(static_cast<double *>(ptr), count);
//     } else {
//         throw std::runtime_error("Unsupported datatype");
//     }
// }
static void init_data(void *ptr, ncclDataType_t datatype, size_t count) {
    if (datatype == ncclInt8) {
        iota(static_cast<int8_t *>(ptr), count);
    } else if (datatype == ncclUint8) {
        iota(static_cast<uint8_t *>(ptr), count);
    } else if (datatype == ncclInt32) {
        iota(static_cast<int32_t *>(ptr), count);
    } else if (datatype == ncclUint32) {
        iota(static_cast<uint32_t *>(ptr), count);
    } else if (datatype == ncclInt64) {
        iota(static_cast<int64_t *>(ptr), count);
    } else if (datatype == ncclUint64) {
        iota(static_cast<uint64_t *>(ptr), count);
    } else if (datatype == ncclHalf) {
#if defined(_Float16)
        iota(static_cast<_Float16 *>(ptr), count);
#else
        // Fill with non-sense values
        iota(static_cast<uint16_t *>(ptr), count);
#endif
    }
#if defined(NCCL_BF16_TYPES_EXIST)
    else if (datatype == ncclBfloat16) {
#if defined(_Float16)
        iota(static_cast<_Float16 *>(ptr), count);
#else
        // Fill with non-sense values
        iota(static_cast<uint16_t *>(ptr), count);
#endif
    }
#endif
    else if (datatype == ncclFloat) {
        iota(static_cast<float *>(ptr), count);
    } else if (datatype == ncclDouble) {
        iota(static_cast<double *>(ptr), count);
    } else {
        throw std::runtime_error("Unsupported datatype");
    }
}

template <typename T> constexpr auto to_GB(const T bytes) -> double {
    constexpr auto GB = static_cast<double>(1 << 30);
    return static_cast<double>(bytes) / GB;
}
} // namespace ncclbench::utils