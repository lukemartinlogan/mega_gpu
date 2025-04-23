cmake_minimum_required(VERSION 3.24)
project(MegaGPU LANGUAGES C CXX)

# ------------------------------------------------------------------------------
# Setup CMake Environment
# ------------------------------------------------------------------------------
set(HSHM_EXPORTED_TARGETS "HermesShm")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
        ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all Executables.")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
        ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all Libraries")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
        ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all static libraries.")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Enable cuda
hshm_enable_cuda(17)

# add_cuda_library(simple_lib STATIC TRUE simple_lib.cc)

# add_cuda_library(cufile_api SHARED TRUE cufile_api.cc)
# target_link_libraries(cufile_api PUBLIC hshm::cudacxx)

# add_cuda_executable(test_basic TRUE test_basic.cc)
# add_dependencies(test_basic simple_lib)
# target_link_libraries(test_basic PUBLIC simple_lib)

# add_cuda_executable(test_cufile_isa TRUE test_cufile_isa.cc)
# target_link_libraries(test_cufile_isa PUBLIC hshm::cudacxx cufile)

# add_cuda_executable(test_cufile_shazzadul TRUE test_cufile_shazzadul.cc)
# target_link_libraries(test_cufile_shazzadul PUBLIC hshm::cudacxx)
add_subdirectory(shared)