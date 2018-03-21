# Ullrich Koethe. Copyright (C) 2018. MIT license

cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

##########
# xvigra #
##########

project(xvigra VERSION 0.1 LANGUAGES CXX)

add_library(xvigra INTERFACE)

set(XVIGRA_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")

target_include_directories(xvigra INTERFACE
  $<BUILD_INTERFACE:${XVIGRA_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>)

export(TARGETS xvigra
  FILE "${PROJECT_BINARY_DIR}/xvigraTargets.cmake")

install(TARGETS xvigra EXPORT xvigraTargets)
install(EXPORT xvigraTargets DESTINATION lib/cmake/xvigra)
install(DIRECTORY ${XVIGRA_INCLUDE_DIR}/ DESTINATION include)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

###########
# options #
###########

OPTION(USE_SIMD "use SSE/AVX acceleration" OFF)

set(BUILD_TESTS off CACHE STRING "build xvigra test suite")
set_property(CACHE BUILD_TESTS PROPERTY STRINGS off gtest doctest)

OPTION(BUILD_BENCHMARK "build xvigra benchmark" OFF)

################
# dependencies #
################

set(XVIGRA_EXTERNALS_PATH "${PROJECT_SOURCE_DIR}/externals" CACHE PATH "path to download external dependencies")

find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
target_link_libraries(xvigra INTERFACE xtl xtensor)

if(USE_SIMD)
    find_package(xsimd REQUIRED)
    target_link_libraries(xvigra INTERFACE xsimd)
    target_compile_definitions(xvigra INTERFACE XVIGRA_USE_SIMD XTENSOR_USE_XSIMD)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
        target_compile_options(xvigra INTERFACE "-march=native")
    elseif(MSVC)
        target_compile_options(xvigra INTERFACE "/arch:AVX2")
    endif()
endif()

###########
# testing #
###########

if(BUILD_TESTS)
    if(BUILD_TESTS STREQUAL "gtest")
        find_package(gtest REQUIRED)
        set(TEST_FRAMEWORK gtest)
    else()
        set(DOCTEST_SOURCE_DIR "${XVIGRA_EXTERNALS_PATH}/doctest" CACHE PATH "Path to doctest")
        if(NOT EXISTS "${DOCTEST_SOURCE_DIR}")
            message(STATUS "Downloading doctest to '${DOCTEST_SOURCE_DIR}'")
            find_package(Git REQUIRED)
            execute_process(COMMAND "${GIT_EXECUTABLE}" clone https://github.com/onqtam/doctest "${DOCTEST_SOURCE_DIR}")
        endif()

        add_library(doctest INTERFACE)
        target_include_directories(doctest INTERFACE "${DOCTEST_SOURCE_DIR}/doctest")
        target_compile_definitions(doctest INTERFACE XVIGRA_USE_DOCTEST)
        set(TEST_FRAMEWORK doctest)
    endif()

    enable_testing()
    add_subdirectory(test)
endif()

################
# benchmarking #
################

if(BUILD_BENCHMARK)
    find_package(benchmark REQUIRED)

    add_subdirectory(benchmark)
endif()
