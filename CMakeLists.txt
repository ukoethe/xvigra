# Ullrich Koethe. Copyright (C) 2018. MIT license

cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(xvigra VERSION 0.1 LANGUAGES CXX)

add_library(xvigra INTERFACE)

set(XVIGRA_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(DOCTEST_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/doctest/doctest)

target_include_directories(xvigra INTERFACE
  $<BUILD_INTERFACE:${XVIGRA_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>)

install(DIRECTORY include/ DESTINATION include)

find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)

enable_testing()
add_subdirectory(test)
