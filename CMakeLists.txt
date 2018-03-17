# Ullrich Koethe. Copyright (C) 2018. MIT license

cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

##########
# xvigra #
##########

project(xvigra VERSION 0.1 LANGUAGES CXX)

add_library(xvigra INTERFACE)

set(XVIGRA_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_include_directories(xvigra INTERFACE
  $<BUILD_INTERFACE:${XVIGRA_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>)

install(DIRECTORY include/ DESTINATION include)

################
# dependencies #
################

find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
target_link_libraries(xvigra INTERFACE xtl xtensor)

set(USE_SIMD off CACHE STRING "use SSE/AVX acceleration")
set_property(CACHE USE_SIMD PROPERTY STRINGS off avx avx2 fma fma4 sse sse2 sse3 ssse3 sse4 sse4a sse4.1 sse4.2 xop)
if(USE_SIMD)
    find_package(xsimd REQUIRED)
    target_compile_definitions(xvigra INTERFACE XVIGRA_USE_SIMD XTENSOR_USE_XSIMD)
    target_link_libraries(xvigra INTERFACE xsimd)
    target_compile_options(xvigra INTERFACE -m${USE_SIMD})
endif()

###########
# testing #
###########

set(BUILD_TESTS off CACHE STRING "build xvigra test suite")
set_property(CACHE BUILD_TESTS PROPERTY STRINGS off gtest doctest)

if(BUILD_TESTS)
    if(BUILD_TESTS STREQUAL "gtest")
        find_package(gtest REQUIRED)
    else()
        set(DOCTEST_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/doctest" CACHE INTERNAL "Path to doctest")
        if(NOT EXISTS "${DOCTEST_SOURCE_DIR}")
            message(STATUS "Downloading doctest to '${DOCTEST_SOURCE_DIR}'")
            find_package(Git REQUIRED)
            execute_process(COMMAND "${GIT_EXECUTABLE}" clone https://github.com/onqtam/doctest
                            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external")
        endif()

        add_library(doctest INTERFACE)
        target_include_directories(doctest INTERFACE "${DOCTEST_SOURCE_DIR}/doctest")
        target_compile_definitions(doctest INTERFACE XVIGRA_USE_DOCTEST)
    endif()

    # find_package(xtensor-io REQUIRED)
    add_library(xtensor_io INTERFACE)
    target_include_directories(xtensor_io INTERFACE "/home/ukoethe/conda/envs/xtensor/include"
                                                    "/home/ukoethe/src/xtensor-io/include/")
    target_link_libraries(xtensor_io INTERFACE "/home/ukoethe/conda/envs/xtensor/lib/libOpenImageIO.so")

    enable_testing()
    add_subdirectory(test)
endif()

################
# benchmarking #
################

OPTION(BUILD_BENCHMARK "build xvigra benchmark" OFF)

if(BUILD_BENCHMARK)
    find_package(benchmark REQUIRED)

    add_subdirectory(benchmark)
endif()
