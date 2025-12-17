cmake_minimum_required(VERSION 3.14)

project(libncclbenchExample CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(../cmake/project-is-top-level.cmake)
include(../cmake/folders.cmake)

if (PROJECT_IS_TOP_LEVEL)
  find_package(libncclbench REQUIRED)
endif ()

add_custom_target(run-examples)

add_executable(ncclbench ncclbench.cpp)

target_compile_features(ncclbench PRIVATE cxx_std_20)

target_link_libraries(ncclbench PRIVATE libncclbench::libncclbench)

include(FetchContent)

FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG 0.8.0 # Can be a tag (yaml-cpp-x.x.x), a commit hash, or a branch name (master)
)
FetchContent_MakeAvailable(yaml-cpp)

target_link_libraries(ncclbench PRIVATE yaml-cpp::yaml-cpp) # The library or executable that require yaml-cpp library
