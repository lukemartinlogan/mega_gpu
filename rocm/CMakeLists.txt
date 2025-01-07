cmake_minimum_required(VERSION 3.24)
project(test_lib LANGUAGES C CXX)

#------------------------------------------------------------------------------
# Build Tests
#------------------------------------------------------------------------------
# Compile test_lib.cc into a library
add_library(test_lib SHARED test_lib.cc)
set_source_files_properties(test_lib.cc PROPERTIES LANGUAGE ${GPU_RUNTIME})
target_compile_definitions(test_lib PRIVATE HERMES_ENABLE_ROCM)
target_compile_options(test_lib PRIVATE -fgpu-rdc)
target_link_libraries(test_lib PRIVATE -fgpu-rdc)

# Add test_lib_main
add_executable(test_lib_main test_lib_main.cc)
target_link_libraries(test_lib_main test_lib)
set_source_files_properties(test_lib_main.cc PROPERTIES LANGUAGE ${GPU_RUNTIME})
target_compile_definitions(test_lib_main PRIVATE HERMES_ENABLE_ROCM)

#-----------------------------------------------------------------------------
# Coverage
#-----------------------------------------------------------------------------
if(HERMES_ENABLE_COVERAGE)
    set_coverage_flags(test_memory_rocm_exec)
endif()