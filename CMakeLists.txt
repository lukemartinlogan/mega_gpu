cmake_minimum_required(VERSION 3.24)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "86")
project(MegaGPU LANGUAGES C CXX CUDA)

add_library(simple_lib STATIC simple_lib.cc)
set_source_files_properties(simple_lib.cc PROPERTIES LANGUAGE CUDA)
set_target_properties(simple_lib PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
#        CUDA_RUNTIME_LIBRARY Shared
#        POSITION_INDEPENDENT_CODE ON
)

add_executable(vector_add vector_add.cc)
add_dependencies(vector_add simple_lib)
target_link_libraries(vector_add PUBLIC simple_lib)
set_source_files_properties(vector_add.cc PROPERTIES LANGUAGE CUDA)
set_target_properties(vector_add PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
#        CUDA_RUNTIME_LIBRARY Shared
#        POSITION_INDEPENDENT_CODE ON
)