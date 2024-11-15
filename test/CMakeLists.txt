cmake_minimum_required(VERSION 3.14)

project(ncclbenchTests LANGUAGES CXX)

include(../cmake/project-is-top-level.cmake)
include(../cmake/folders.cmake)

# ---- Dependencies ----

if(PROJECT_IS_TOP_LEVEL)
  find_package(ncclbench REQUIRED)
  enable_testing()
endif()

# ---- Tests ----

add_executable(ncclbench_test source/ncclbench_test.cpp)
target_link_libraries(ncclbench_test PRIVATE ncclbench::ncclbench)
target_compile_features(ncclbench_test PRIVATE cxx_std_17)

add_test(NAME ncclbench_test COMMAND ncclbench_test)

# ---- End-of-file commands ----

add_folders(Test)
