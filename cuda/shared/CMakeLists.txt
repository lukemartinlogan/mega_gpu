cmake_minimum_required(VERSION 3.24)
project(MegaGPU LANGUAGES C CXX)

# ------------------------------------------------------------------------------
# Setup CMake Environment
# ------------------------------------------------------------------------------
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_cuda_library(simple_lib_so SHARED TRUE so_test.cc)
target_link_libraries(simple_lib_so PUBLIC hshm::cudacxx)

# add_library(simple_lib_so SHARED so_test.cc)
# set_source_files_properties(so_test.cc PROPERTIES LANGUAGE CUDA)
# set_target_properties(${LIB_NAME} PROPERTIES
# CUDA_SEPARABLE_COMPILATION ON
# POSITION_INDEPENDENT_CODE ON
# CUDA_RUNTIME_LIBRARY Shared
# )
add_cuda_executable(test_shared TRUE test_shared.cc)
add_dependencies(test_shared simple_lib_so)
target_link_libraries(test_shared PUBLIC simple_lib_so)
