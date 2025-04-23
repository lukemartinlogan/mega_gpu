cmake_minimum_required(VERSION 3.24)
project(MegaGPU LANGUAGES C CXX)

# ------------------------------------------------------------------------------
# Enable CUDA
# ------------------------------------------------------------------------------
# Enable cuda boilerplate
macro(hshm_enable_cuda CXX_STANDARD)
    set(CMAKE_CUDA_STANDARD ${CXX_STANDARD})
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()

    message("USING CUDA ARCH: ${CMAKE_CUDA_ARCHITECTURES}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --forward-unknown-to-host-compiler -diag-suppress=177,20014,20011,20012")
    enable_language(CUDA)

    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)
endmacro()

# Function for setting source files for cuda
function(set_cuda_sources DO_COPY SRC_FILES CUDA_SOURCE_FILES_VAR)
    set(CUDA_SOURCE_FILES ${${CUDA_SOURCE_FILES_VAR}} PARENT_SCOPE)

    foreach(SOURCE IN LISTS SRC_FILES)
        if(${DO_COPY})
            set(CUDA_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/cuda/${SOURCE})
            configure_file(${SOURCE} ${CUDA_SOURCE} COPYONLY)
        else()
            set(CUDA_SOURCE ${SOURCE})
        endif()

        list(APPEND CUDA_SOURCE_FILES ${CUDA_SOURCE})
        set_source_files_properties(${CUDA_SOURCE} PROPERTIES LANGUAGE CUDA)
    endforeach()

    set(${CUDA_SOURCE_FILES_VAR} ${CUDA_SOURCE_FILES} PARENT_SCOPE)
endfunction()

# Function for adding a CUDA library
function(add_cuda_library TARGET SHARED DO_COPY)
    set(SRC_FILES ${ARGN})
    set(CUDA_SOURCE_FILES "")
    set_cuda_sources("${DO_COPY}" "${SRC_FILES}" CUDA_SOURCE_FILES)
    add_library(${TARGET} ${SHARED} ${CUDA_SOURCE_FILES})
    target_compile_options(${TARGET} PUBLIC
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

    if(SHARED STREQUAL "SHARED")
        set_target_properties(${TARGET} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON
            CUDA_RUNTIME_LIBRARY Shared
        )
    else()
        set_target_properties(${TARGET} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON
            CUDA_RUNTIME_LIBRARY Static
        )
    endif()
endfunction()

# Function for adding a CUDA executable
function(add_cuda_executable TARGET DO_COPY)
    set(SRC_FILES ${ARGN})
    set(CUDA_SOURCE_FILES "")
    set_cuda_sources("${DO_COPY}" "${SRC_FILES}" CUDA_SOURCE_FILES)
    add_executable(${TARGET} ${CUDA_SOURCE_FILES})
    set_target_properties(${TARGET} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
    target_compile_options(${TARGET} PUBLIC
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endfunction()

# ------------------------------------------------------------------------------
# Setup CMake Environment
# ------------------------------------------------------------------------------
hshm_enable_cuda(17)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_cuda_executable(test_shared TRUE test_shared.cc)
