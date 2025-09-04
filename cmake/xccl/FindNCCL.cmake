# FindNCCL.cmake - Modern CMake module to locate NCCL
#
# Variables set:
#   NCCL_FOUND           - True if NCCL was found
#   NCCL_INCLUDE_DIRS    - Path(s) to NCCL headers
#   NCCL_LIBRARIES       - Path to NCCL library
#
# Targets created:
#   NCCL::NCCL

cmake_minimum_required(VERSION 3.12)

# Allow user hints via environment
set(_NCCL_HINTS "")
if(DEFINED ENV{NCCL_ROOT})
  list(APPEND _NCCL_HINTS $ENV{NCCL_ROOT})
endif()
if(DEFINED ENV{NCCL_INCLUDE_DIR})
  list(APPEND _NCCL_HINTS $ENV{NCCL_INCLUDE_DIR})
endif()
if(DEFINED ENV{NCCL_LIB_DIR})
  list(APPEND _NCCL_HINTS $ENV{NCCL_LIB_DIR})
endif()
if(DEFINED CUDA_TOOLKIT_ROOT_DIR)
  list(APPEND _NCCL_HINTS ${CUDA_TOOLKIT_ROOT_DIR})
endif()

# Find headers
find_path(NCCL_INCLUDE_DIRS
  NAMES nccl.h
  HINTS ${_NCCL_HINTS}
  PATH_SUFFIXES include)

# Find library (prefer static if USE_STATIC_NCCL=ON)
set(_NCCL_NAMES nccl)
if(USE_STATIC_NCCL)
  list(INSERT _NCCL_NAMES 0 nccl_static)
endif()

find_library(NCCL_LIBRARIES
  NAMES ${_NCCL_NAMES}
  HINTS ${_NCCL_HINTS}
  PATH_SUFFIXES lib lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL
  REQUIRED_VARS NCCL_INCLUDE_DIRS NCCL_LIBRARIES)

if(NCCL_FOUND)
  add_library(NCCL::NCCL UNKNOWN IMPORTED)
  set_target_properties(NCCL::NCCL PROPERTIES
    IMPORTED_LOCATION "${NCCL_LIBRARIES}"
    INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIRS}")

  # If using static NCCL, add required system libs
  if(NCCL_LIBRARIES MATCHES ".*nccl_static.*")
    find_package(Threads REQUIRED)
    set_property(TARGET NCCL::NCCL APPEND PROPERTY
      INTERFACE_LINK_LIBRARIES
        ${CMAKE_DL_LIBS}
        Threads::Threads
    )
  endif()
endif()