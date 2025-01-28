cmake_minimum_required(VERSION 3.24)
project(mega_gpu LANGUAGES C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ------------------------------------------------------------------------------
# DOTENV
# ------------------------------------------------------------------------------
if(EXISTS ${CMAKE_SOURCE_DIR}/.env.cmake)
    include(.env.cmake)
endif()


#------------------------------------------------------------------------------
# CMake Options
#------------------------------------------------------------------------------

option(DO_CUDA "Enable CUDA" OFF)
option(DO_ROCM "Enable ROCm" OFF)
option(BUILD_SHARED_LIBS "Build as a shared library" OFF)

#------------------------------------------------------------------------------
# Setup CMake Environment
#------------------------------------------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
        ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all Executables.")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
        ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all Libraries")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
        ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all static libraries.")

# This is for compatability with CLANG + Spack
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# HermesShm
find_package(HermesShm CONFIG REQUIRED)

#------------------------------------------------------------------------------
# Add sources
#------------------------------------------------------------------------------
if (DO_CUDA)
    add_subdirectory(cuda)
endif()

if (DO_ROCM)
    add_subdirectory(rocm)
endif()

