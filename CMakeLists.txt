# Ullrich Koethe. Copyright (C) 2018. MIT license

cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(xvigra VERSION 0.1 LANGUAGES CXX)

add_library(xvigra INTERFACE)

set(XVIGRA_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_include_directories(xvigra INTERFACE
  $<BUILD_INTERFACE:${XVIGRA_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>)

install(DIRECTORY include/ DESTINATION include)

find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)

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

    enable_testing()
    add_subdirectory(test)
endif()

