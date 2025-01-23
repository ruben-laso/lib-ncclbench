cmake_minimum_required(VERSION 3.14)

project(libncclbenchTests LANGUAGES CXX)

include(../cmake/project-is-top-level.cmake)
include(../cmake/folders.cmake)

# ---- Dependencies ----

if(PROJECT_IS_TOP_LEVEL)
  find_package(libncclbench REQUIRED)
  enable_testing()
endif()

# ---- Tests ----

add_executable(libncclbench_test source/libncclbench_test.cpp)
target_link_libraries(libncclbench_test PRIVATE libncclbench::libncclbench)
target_compile_features(libncclbench_test PRIVATE cxx_std_17)

add_test(NAME libncclbench_test COMMAND libncclbench_test)

# ---- End-of-file commands ----

add_folders(Test)
