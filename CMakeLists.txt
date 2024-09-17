cmake_minimum_required(VERSION 3.24)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "60;70;80")
project(MegaGPU LANGUAGES C CXX CUDA)

add_executable(vector_add vector_add.cpp)
set_source_files_properties(vector_add.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(vector_add PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)