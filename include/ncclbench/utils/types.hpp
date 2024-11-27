#pragma once

#include <map>
#include <stdexcept>

#include <mpi.h>

#include "../xccl.hpp"

namespace ncclbench::types {

inline auto mpi_to_nccl(const MPI_Datatype datatype) -> ncclDataType_t {
    std::map<MPI_Datatype, ncclDataType_t> typeMap = {{MPI_BYTE, ncclInt8},
                                                      {MPI_INT, ncclInt32},
                                                      {MPI_DOUBLE, ncclFloat64},
                                                      {MPI_CHAR, ncclChar},
                                                      {MPI_FLOAT, ncclFloat32}};
    const auto it = typeMap.find(datatype);
    if (it == typeMap.end()) {
        throw std::runtime_error("Unsupported datatype");
    }
    return it->second;
}

inline MPI_Datatype str_to_mpi(const std::string_view &str) {
    if (str == "byte") {
        return MPI_BYTE;
    }
    if (str == "int") {
        return MPI_INT;
    }
    if (str == "double") {
        return MPI_DOUBLE;
    }
    if (str == "char") {
        return MPI_CHAR;
    }
    if (str == "float") {
        return MPI_FLOAT;
    }

    throw std::runtime_error("Unsupported datatype");
}

inline ncclDataType_t str_to_nccl(const std::string_view &str) {
    return mpi_to_nccl(str_to_mpi(str));
}

inline auto size_of(const MPI_Datatype datatype) -> size_t {
    int size;
    MPI_Type_size(datatype, &size);
    return static_cast<size_t>(size);
}

inline auto bytes_to_elements(const size_t bytes, const MPI_Datatype datatype)
    -> size_t {
    return bytes / size_of(datatype);
}

} // namespace ncclbench::types
