cmake_minimum_required(VERSION 3.18)

set(PROJECT_NAME nanoeigenpy)
set(PROJECT_URL https://github.com/ManifoldFR/nanoeigenpy)
set(PROJECT_DESCRIPTION "Tools for using Eigen with nanobind")
set(PROJECT_CUSTOM_HEADER_EXTENSION "hpp")
set(PROJECT_USE_CMAKE_EXPORT True)

include(cmake/base.cmake)

COMPUTE_PROJECT_ARGS(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

find_package(Eigen3 REQUIRED)

add_library(nanoeigenpy_headers INTERFACE)
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

file(GLOB nanoeigenpy_SOURCES src/*.cpp)
nanobind_add_module(nanoeigenpy NB_STATIC NB_SUPPRESS_WARNINGS ${nanoeigenpy_SOURCES})
target_link_libraries(nanoeigenpy PUBLIC nanoeigenpy_headers)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

install(
  TARGETS nanoeigenpy_headers
  EXPORT ${TARGETS_EXPORT_NAME}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
  DIRECTORY include/nanoeigenpy
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "*.hpp"
)

install(
  TARGETS nanoeigenpy
  EXPORT ${TARGETS_EXPORT_NAME}
  LIBRARY DESTINATION ${Python_SITELIB}
)
