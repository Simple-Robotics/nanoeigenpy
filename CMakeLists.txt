#
# Copyright 2025 INRIA
#

cmake_minimum_required(VERSION 3.18)

set(PROJECT_NAME nanoeigenpy)
set(PROJECT_URL https://github.com/Simple-Robotics/nanoeigenpy)
set(PROJECT_DESCRIPTION "Tools for using Eigen with nanobind")
set(PROJECT_CUSTOM_HEADER_EXTENSION "hpp")
set(PROJECT_USE_CMAKE_EXPORT True)

include(cmake/base.cmake)
include(cmake/ide.cmake)

COMPUTE_PROJECT_ARGS(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(
    CACHE CMAKE_BUILD_TYPE
    PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo"
  )
endif()

# Find dependencies
ADD_PROJECT_DEPENDENCY(Eigen3 REQUIRED PKG_CONFIG_REQUIRES "eigen3 >= 3.0.5")

find_package(Python REQUIRED COMPONENTS Interpreter Development)

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT
)
ADD_PROJECT_DEPENDENCY(nanobind CONFIG REQUIRED)

# Setup main targets
set(
  ${PROJECT_NAME}_HEADERS
  include/nanoeigenpy/computation-info.hpp
  include/nanoeigenpy/computation-info.hpp
  include/nanoeigenpy/decompositions.hpp
  include/nanoeigenpy/eigen-typedef.hpp
  include/nanoeigenpy/fwd.hpp
  include/nanoeigenpy/geometry.hpp
)

add_library(nanoeigenpy_headers INTERFACE)
target_include_directories(
  nanoeigenpy_headers
  INTERFACE
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(nanoeigenpy_headers INTERFACE Eigen3::Eigen)

set(PROJECT_HEADERS_FULL_PATH)
foreach(header ${${PROJECT_NAME}_HEADERS})
  list(APPEND PROJECT_HEADERS_FULL_PATH ${PROJECT_SOURCE_DIR}/${header})
endforeach()

file(GLOB ${PROJECT_NAME}_SOURCES src/*.cpp)
nanobind_add_module(nanoeigenpy NB_STATIC NB_SUPPRESS_WARNINGS ${nanoeigenpy_SOURCES} ${PROJECT_HEADERS_FULL_PATH})
target_link_libraries(nanoeigenpy PUBLIC nanoeigenpy_headers)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

# Install targets
install(
  TARGETS ${PROJECT_NAME}_headers
  EXPORT ${TARGETS_EXPORT_NAME}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${TARGETS_EXPORT_NAME}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

ADD_HEADER_GROUP(${PROJECT_NAME}_HEADERS)
ADD_SOURCE_GROUP(${PROJECT_NAME}_SOURCES)
