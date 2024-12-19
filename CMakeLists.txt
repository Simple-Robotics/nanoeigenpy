cmake_minimum_required(VERSION 3.18)

set(PROJECT_NAME nanoeigenpy)
set(PROJECT_URL https://github.com/ManifoldFR/nanoeigenpy)
set(PROJECT_DESCRIPTION "Tools for using Eigen with nanobind")
set(PROJECT_CUSTOM_HEADER_EXTENSION "hpp")

include(cmake/base.cmake)

COMPUTE_PROJECT_ARGS(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})

find_package(Eigen3 REQUIRED)

add_library(nanoeigenpy_headers INTERFACE)
target_include_directories(nanoeigenpy_headers INTERFACE ./src)
target_link_libraries(nanoeigenpy_headers INTERFACE Eigen3::Eigen)

find_package(Python REQUIRED COMPONENTS Interpreter Development)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(
    CACHE CMAKE_BUILD_TYPE
    PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo"
  )
endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED)

set(nanoeigenpy_SOURCES src/nanoeigenpy/module.cpp)
nanobind_add_module(nanoeigenpy ${nanoeigenpy_SOURCES})
target_link_libraries(nanoeigenpy PUBLIC nanoeigenpy_headers)
