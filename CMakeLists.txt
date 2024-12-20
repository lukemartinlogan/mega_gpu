cmake_minimum_required(VERSION 3.24)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)
project(MegaGPU LANGUAGES C CXX CUDA)

# HermesShm
find_package(HermesShm CONFIG REQUIRED)
message(STATUS "found hermes_shm.h at ${HermesShm_INCLUDE_DIRS}")
include_directories(${HermesShm_INCLUDE_DIRS})
link_directories(${HermesShm_LIBRARY_DIRS})

add_library(simple_lib STATIC simple_lib.cc)
set_source_files_properties(simple_lib.cc PROPERTIES LANGUAGE CUDA)
set_target_properties(simple_lib PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
#        CUDA_RUNTIME_LIBRARY Shared
#        POSITION_INDEPENDENT_CODE ON
)

add_library(cufile_api SHARED cufile_api.cc)
target_link_libraries(cufile_api PUBLIC ${HermesShm_LIBRARIES})
set_source_files_properties(cufile_api.cc PROPERTIES LANGUAGE CUDA)
set_target_properties(cufile_api PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
       CUDA_RUNTIME_LIBRARY Shared
       POSITION_INDEPENDENT_CODE ON
)

add_executable(test_basic test_basic.cc)
add_dependencies(test_basic simple_lib)
target_link_libraries(test_basic PUBLIC simple_lib)
set_source_files_properties(test_basic.cc PROPERTIES LANGUAGE CUDA)
set_target_properties(test_basic PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
#        CUDA_RUNTIME_LIBRARY Shared
#        POSITION_INDEPENDENT_CODE ON
)

add_executable(test_cufile_isa test_cufile_isa.cc)
target_link_libraries(test_cufile_isa PUBLIC simple_lib ${HermesShm_LIBRARIES} cufile_api cufile )
set_source_files_properties(test_cufile_isa.cc PROPERTIES LANGUAGE CUDA)
set_target_properties(test_cufile_isa PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
#        CUDA_RUNTIME_LIBRARY Shared
#        POSITION_INDEPENDENT_CODE ON
)

add_executable(test_cufile_shazzadul test_cufile_shazzadul.cc)
target_link_libraries(test_cufile_shazzadul PUBLIC simple_lib ${HermesShm_LIBRARIES})
set_source_files_properties(test_cufile_shazzadul.cc PROPERTIES LANGUAGE CUDA)
set_target_properties(test_cufile_shazzadul PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
#        CUDA_RUNTIME_LIBRARY Shared
#        POSITION_INDEPENDENT_CODE ON
)


