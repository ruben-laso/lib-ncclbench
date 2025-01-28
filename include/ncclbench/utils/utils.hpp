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

static void init_data(void *ptr, MPI_Datatype datatype, size_t count) {
    if (datatype == MPI_BYTE) {
        iota(static_cast<char *>(ptr), count);
    } else if (datatype == MPI_CHAR) {
        iota(static_cast<char *>(ptr), count);
    } else if (datatype == MPI_INT) {
        iota(static_cast<int *>(ptr), count);
    } else if (datatype == MPI_FLOAT) {
        iota(static_cast<float *>(ptr), count);
    } else if (datatype == MPI_DOUBLE) {
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