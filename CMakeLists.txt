cmake_minimum_required(VERSION 3.14)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/xccl")

include(cmake/prelude.cmake)

project(
    libncclbench
    VERSION 0.1.0
    DESCRIPTION "Library for benchmarking NCCL"
    HOMEPAGE_URL "https://github.com/ruben-laso/lib-ncclbench"
    LANGUAGES CXX CUDA
)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(cmake/project-is-top-level.cmake)
include(cmake/variables.cmake)

# ---- Declare options ----

set(XCCL "NCCL" CACHE STRING "The library to benchmark: NCCL (default) or RCCL")
set_property(CACHE XCCL PROPERTY STRINGS "NCCL" "RCCL")
if (NOT XCCL STREQUAL "NCCL" AND NOT XCCL STREQUAL "RCCL")
  message(FATAL_ERROR "Invalid value for XCCL: ${XCCL}")
endif ()
if (XCCL STREQUAL "RCCL")
  add_definitions(-DUSE_RCCL)
endif ()

message(STATUS "Using ${XCCL} as the library to benchmark")

# ---- Declare sources ----

list(APPEND REL_SOURCES
    "source/ncclbench.cpp"
    "source/benchmarks/nccl_allgather.cpp"
    "source/benchmarks/nccl_allreduce.cpp"
    "source/benchmarks/nccl_alltoall.cpp"
    "source/benchmarks/nccl_broadcast.cpp"
    "source/benchmarks/nccl_p2p.cpp"
    "source/benchmarks/nccl_reduce.cpp"
    "source/benchmarks/nccl_reduce_scatter.cpp"
)
foreach (SOURCE IN LISTS REL_SOURCES)
  get_filename_component(SOURCE_PATH "${SOURCE}" ABSOLUTE)
  list(APPEND SOURCES "${SOURCE_PATH}")
endforeach ()

list(APPEND REL_HEADERS
    "include/ncclbench/ncclbench.hpp"
    "include/ncclbench/xccl.hpp"
    "include/ncclbench/benchmarks/benchmarks.hpp"
    "include/ncclbench/benchmarks/common.hpp"
    "include/ncclbench/benchmarks/nccl_template.hpp"
    "include/ncclbench/utils/checks.hpp"
    "include/ncclbench/utils/host.hpp"
    "include/ncclbench/utils/nccl_functions.hpp"
    "include/ncclbench/utils/types.hpp"
    "include/ncclbench/utils/utils.hpp"
)
foreach (HEADER IN LISTS REL_HEADERS)
  get_filename_component(HEADER_PATH "${HEADER}" ABSOLUTE)
  list(APPEND HEADERS "${HEADER_PATH}")
endforeach ()

message(STATUS "Sources:")
foreach (SOURCE IN LISTS SOURCES)
  file(RELATIVE_PATH REL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}" "${SOURCE}")
  message(STATUS "  ${REL_SOURCE}")
endforeach ()
foreach (HEADER IN LISTS HEADERS)
  file(RELATIVE_PATH REL_HEADER "${CMAKE_CURRENT_SOURCE_DIR}" "${HEADER}")
  message(STATUS "  ${REL_HEADER}")
endforeach ()

if (XCCL STREQUAL "RCCL")
  # For each CUDA source file, hipify it and add it to the SOURCES list
  include(Hipify)
  foreach (SOURCE ${SOURCES})
    hipify(${SOURCE})
    list(APPEND HIPIFIED_SOURCES ${LAST_HIP_FILE})
  endforeach ()
  foreach (HEADER ${HEADERS})
    hipify(${HEADER})
    list(APPEND HIPIFIED_SOURCES ${LAST_HIP_FILE})
  endforeach ()
  set(SOURCES ${HIPIFIED_SOURCES})
endif ()

# ---- Dependencies ----

find_package(MPI REQUIRED)
list(APPEND LINK_LIBRARIES MPI::MPI_CXX)

if (XCCL STREQUAL "NCCL")
  find_package(CUDAToolkit REQUIRED)
  list(APPEND LINK_LIBRARIES CUDA::cudart)

  find_package(NCCL REQUIRED)  # Only call find_package if paths are not provided
  list(APPEND LINK_LIBRARIES NCCL::NCCL)
elseif (XCCL STREQUAL "RCCL")
  message(FATAL_ERROR "RCCL is not supported yet")
endif ()

# ---- Declare library ----

add_library(libncclbench_libncclbench ${SOURCES})
if (XCCL STREQUAL "NCCL")
  set_target_properties(libncclbench_libncclbench PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
  )
endif ()

add_library(libncclbench::libncclbench ALIAS libncclbench_libncclbench)

include(GenerateExportHeader)
generate_export_header(
    libncclbench_libncclbench
    BASE_NAME ncclbench
    EXPORT_FILE_NAME export/ncclbench/ncclbench_export.hpp
    CUSTOM_CONTENT_FROM_VARIABLE pragma_suppress_c4251
)

if (NOT BUILD_SHARED_LIBS)
  target_compile_definitions(libncclbench_libncclbench PUBLIC LIBNCCLBENCH_STATIC_DEFINE)
endif ()

set_target_properties(
    libncclbench_libncclbench PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    VERSION "${PROJECT_VERSION}"
    SOVERSION "${PROJECT_VERSION_MAJOR}"
    EXPORT_NAME libncclbench
    OUTPUT_NAME ncclbench
)

target_include_directories(
    libncclbench_libncclbench ${warning_guard}
    PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
)

target_include_directories(
    libncclbench_libncclbench SYSTEM
    PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/export>"
)

target_compile_features(libncclbench_libncclbench PUBLIC cxx_std_17)

# ---- Link dependencies ----

foreach (LIBRARY IN ITEMS ${LINK_LIBRARIES})
  message(STATUS "Linking to library: ${LIBRARY}")
  target_link_libraries(libncclbench_libncclbench ${LIBRARY})
endforeach ()

foreach (INCLUDE_DIR IN ITEMS ${INCLUDE_DIRS})
  message(STATUS "Including directory: ${INCLUDE_DIR}")
  target_include_directories(libncclbench_libncclbench PUBLIC ${INCLUDE_DIR})
endforeach ()

# ---- Install rules ----

if (NOT CMAKE_SKIP_INSTALL_RULES)
  include(cmake/install-rules.cmake)
endif ()

# ---- Examples ----

if (PROJECT_IS_TOP_LEVEL)
  option(BUILD_EXAMPLES "Build examples tree." "${libncclbench_DEVELOPER_MODE}")
  if (BUILD_EXAMPLES)
    add_subdirectory(example)
  endif ()
endif ()

# ---- Developer mode ----

if (NOT libncclbench_DEVELOPER_MODE)
  return()
elseif (NOT PROJECT_IS_TOP_LEVEL)
  message(
      AUTHOR_WARNING
      "Developer mode is intended for developers of libncclbench"
  )
endif ()

include(cmake/dev-mode.cmake)
