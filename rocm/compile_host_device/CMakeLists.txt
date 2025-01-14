project(static_lib)

cmake_minimum_required(VERSION 3.10)

set(GPU_RUNTIME "HIP" CACHE STRING "Switches between HIP and CUDA")
set(GPU_RUNTIMES "HIP" "CUDA")
set_property(CACHE GPU_RUNTIME PROPERTY STRINGS ${GPU_RUNTIMES})
set(GPU_RUNTIME "HIP" CACHE STRING "Switches between HIP and CUDA")
if(NOT "${GPU_RUNTIME}" IN_LIST GPU_RUNTIMES)
    set(ERROR_MESSAGE
        "GPU_RUNTIME is set to \"${GPU_RUNTIME}\".\nGPU_RUNTIME must be either HIP or CUDA."
    )
    message(FATAL_ERROR ${ERROR_MESSAGE})
endif()
enable_language(${GPU_RUNTIME})
set(CMAKE_${GPU_RUNTIME}_STANDARD 17)
set(CMAKE_${GPU_RUNTIME}_EXTENSIONS OFF)
set(CMAKE_${GPU_RUNTIME}_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --forward-unknown-to-host-compiler")
set(ROCM_ROOT
        "/opt/rocm"
        CACHE PATH
        "Root directory of the ROCm installation"
)
if(GPU_RUNTIME STREQUAL "CUDA")
    list(APPEND include_dirs "${ROCM_ROOT}/include")
endif()
find_package(HIP REQUIRED)
message(STATUS "found hip at ${HIP_DIR}")

# For windows, AR is MS Librarian and that is picked by Visual Studio's command prompt.
if (WIN32)
  find_program(libpath NAMES lib.exe)
  set (CMAKE_AR ${libpath})
endif()

# Set compiler and linker
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_BUILD_TYPE Release)

# Turn static library generation ON
option(BUILD_SHARED_LIBS "Build as a shared library" OFF)

# For windows, We need to tell cmake how to create static library.
if (WIN32)
  set (CMAKE_CXX_CREATE_STATIC_LIBRARY "<CMAKE_AR> /out:<TARGET> <LINK_FLAGS> <OBJECTS>")
endif()

# Generate static host+gpu lib libHipDevice.a
set(CPP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_lib.cc)
add_library(HipDevice STATIC ${CPP_SOURCES})
target_compile_options(HipDevice PRIVATE -fgpu-rdc)
target_link_libraries(HipDevice PRIVATE -fgpu-rdc)
target_include_directories(HipDevice PRIVATE ${CMAKE_PREFIX_PATH}/hsa/include)
set_source_files_properties(${CPP_SOURCES} PROPERTIES LANGUAGE ${GPU_RUNTIME})

# Generate static host lib libHipDevice2.a
set(CPP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_lib2.cc)
add_library(HipDevice2 SHARED ${CPP_SOURCES})
target_compile_options(HipDevice2 PRIVATE -fgpu-rdc)
target_link_libraries(HipDevice2 PRIVATE -fgpu-rdc HipDevice)
target_include_directories(HipDevice2 PRIVATE ${CMAKE_PREFIX_PATH}/hsa/include)
set_source_files_properties(${CPP_SOURCES} PROPERTIES LANGUAGE ${GPU_RUNTIME})

# Create test executable that uses libHipDevice.a
set(TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_lib_main.cc)
add_executable(test_device_static ${TEST_SOURCES})
add_dependencies(test_device_static HipDevice)
target_compile_options(test_device_static PRIVATE -fgpu-rdc)
set_source_files_properties(${TEST_SOURCES} PROPERTIES LANGUAGE ${GPU_RUNTIME})
if (WIN32)
  target_link_libraries(test_device_static PRIVATE -lHipDevice -L${CMAKE_CURRENT_BINARY_DIR})
else()
  target_link_libraries(test_device_static PRIVATE HipDevice HipDevice2)
endif()
target_link_libraries(test_device_static PRIVATE -fgpu-rdc amdhip64 amd_comgr)