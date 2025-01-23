cmake_minimum_required(VERSION 3.14)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/xccl")

include(cmake/prelude.cmake)

project(
    ncclbench
    VERSION 0.1.0
    DESCRIPTION "Library for benchmarking NCCL"
    HOMEPAGE_URL "https://github.com/ruben-laso/lib-ncclbench"
    LANGUAGES CXX
)

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

set(NCCL_HOME "" CACHE PATH "Path to the NCCL installation directory")
if (NCCL_HOME)
  message(STATUS "Using NCCL from ${NCCL_HOME}")
endif ()

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
  # CUDA
  find_package(CUDA REQUIRED)
  list(APPEND LINK_LIBRARIES ${CUDA_LIBRARIES})

  find_package(NCCL REQUIRED)  # Only call find_package if paths are not provided

  # if artifact "nccl" exists, add it to LINK_LIBRARIES
  if (TARGET nccl)
    list(APPEND LINK_LIBRARIES nccl)  # Ensure 'nccl' is linked
  else ()
    list(APPEND INCLUDE_DIRS ${NCCL_INCLUDE_DIRS})
    list(APPEND LINK_LIBRARIES ${NCCL_LIBRARIES})
  endif ()
elseif (XCCL STREQUAL "RCCL")
  message(FATAL_ERROR "RCCL is not supported yet")
endif ()

# ---- Declare library ----

if (XCCL STREQUAL "NCCL")
  cuda_add_library(ncclbench_ncclbench ${SOURCES})
else ()
  add_library(ncclbench_ncclbench ${SOURCES})
endif ()

add_library(ncclbench::ncclbench ALIAS ncclbench_ncclbench)

include(GenerateExportHeader)
generate_export_header(
    ncclbench_ncclbench
    BASE_NAME ncclbench
    EXPORT_FILE_NAME export/ncclbench/ncclbench_export.hpp
    CUSTOM_CONTENT_FROM_VARIABLE pragma_suppress_c4251
)

if (NOT BUILD_SHARED_LIBS)
  target_compile_definitions(ncclbench_ncclbench PUBLIC NCCLBENCH_STATIC_DEFINE)
endif ()

set_target_properties(
    ncclbench_ncclbench PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    VERSION "${PROJECT_VERSION}"
    SOVERSION "${PROJECT_VERSION_MAJOR}"
    EXPORT_NAME ncclbench
    OUTPUT_NAME ncclbench
)

target_include_directories(
    ncclbench_ncclbench ${warning_guard}
    PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
)

target_include_directories(
    ncclbench_ncclbench SYSTEM
    PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/export>"
)

target_compile_features(ncclbench_ncclbench PUBLIC cxx_std_17)

# ---- Link dependencies ----

foreach (LIBRARY IN ITEMS ${LINK_LIBRARIES})
  message(STATUS "Linking to library: ${LIBRARY}")
  target_link_libraries(ncclbench_ncclbench ${LIBRARY})
endforeach ()

foreach (INCLUDE_DIR IN ITEMS ${INCLUDE_DIRS})
  message(STATUS "Including directory: ${INCLUDE_DIR}")
  target_include_directories(ncclbench_ncclbench PUBLIC ${INCLUDE_DIR})
endforeach ()

# ---- Install rules ----

if (NOT CMAKE_SKIP_INSTALL_RULES)
  include(cmake/install-rules.cmake)
endif ()

# ---- Examples ----

if (PROJECT_IS_TOP_LEVEL)
  option(BUILD_EXAMPLES "Build examples tree." "${ncclbench_DEVELOPER_MODE}")
  if (BUILD_EXAMPLES)
    add_subdirectory(example)
  endif ()
endif ()

# ---- Developer mode ----

if (NOT ncclbench_DEVELOPER_MODE)
  return()
elseif (NOT PROJECT_IS_TOP_LEVEL)
  message(
      AUTHOR_WARNING
      "Developer mode is intended for developers of ncclbench"
  )
endif ()

include(cmake/dev-mode.cmake)
