#
# Copyright 2025 INRIA
#

cmake_minimum_required(VERSION 3.18)

set(PROJECT_NAME nanoeigenpy)
set(PROJECT_URL https://github.com/Simple-Robotics/nanoeigenpy)
set(
  PROJECT_DESCRIPTION
  "A support library for bindings between Eigen in C++ and Python, based on nanobind"
)
set(PROJECT_CUSTOM_HEADER_EXTENSION "hpp")
set(PROJECT_USE_CMAKE_EXPORT True)

# To enable jrl-cmakemodules compatibility with workspace we must define the two
# following lines
set(PROJECT_AUTO_RUN_FINALIZE FALSE)
set(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})

# Check if the submodule cmake have been initialized
set(JRL_CMAKE_MODULES "${CMAKE_CURRENT_LIST_DIR}/cmake")
if(EXISTS "${JRL_CMAKE_MODULES}/base.cmake")
  message(STATUS "JRL cmakemodules found in 'cmake/' git submodule")
else()
  find_package(jrl-cmakemodules QUIET CONFIG)
  if(jrl-cmakemodules_FOUND)
    get_property(
      JRL_CMAKE_MODULES
      TARGET jrl-cmakemodules::jrl-cmakemodules
      PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    )
    message(STATUS "JRL cmakemodules found on system at ${JRL_CMAKE_MODULES}")
  elseif(${CMAKE_VERSION} VERSION_LESS "3.14.0")
    message(
      FATAL_ERROR
      "\nCan't find jrl-cmakemodules. Please either:\n"
      "  - use git submodule: 'git submodule update --init'\n"
      "  - or install https://github.com/jrl-umi3218/jrl-cmakemodules\n"
      "  - or upgrade your CMake version to >= 3.14 to allow automatic fetching\n"
    )
  else()
    message(STATUS "JRL cmakemodules not found. Let's fetch it.")
    include(FetchContent)
    FetchContent_Declare(
      "jrl-cmakemodules"
      GIT_REPOSITORY "https://github.com/jrl-umi3218/jrl-cmakemodules.git"
    )
    FetchContent_MakeAvailable("jrl-cmakemodules")
    FetchContent_GetProperties("jrl-cmakemodules" SOURCE_DIR JRL_CMAKE_MODULES)
  endif()
endif()

option(INSTALL_DOCUMENTATION "Generate and install the documentation" OFF)

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0167 NEW)
endif()
if(POLICY CMP0177)
  cmake_policy(SET CMP0177 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0177 NEW)
endif()
include("${JRL_CMAKE_MODULES}/base.cmake")
COMPUTE_PROJECT_ARGS(PROJECT_ARGS LANGUAGES CXX)
include("${JRL_CMAKE_MODULES}/ide.cmake")
include("${JRL_CMAKE_MODULES}/apple.cmake")
project(${PROJECT_NAME} ${PROJECT_ARGS})

string(REPLACE "-pedantic" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
string(REPLACE "-Wcast-qual" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
string(REPLACE "-Wconversion" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(
    CACHE CMAKE_BUILD_TYPE
    PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo"
  )
endif()

option(
  BUILD_WITH_CHOLMOD_SUPPORT
  "Build NanoEigenPy with the Cholmod support"
  OFF
)

if(APPLE)
  option(
    BUILD_WITH_ACCELERATE_SUPPORT
    "Build EigenPy with the Accelerate support"
    OFF
  )
endif(APPLE)

# Find dependencies
ADD_PROJECT_DEPENDENCY(Eigen3 REQUIRED PKG_CONFIG_REQUIRES "eigen3 >= 3.3.1")

find_package(Python REQUIRED COMPONENTS Interpreter Development)

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED)

# Setup main targets
file(
  GLOB_RECURSE ${PROJECT_NAME}_HEADERS
  CONFIGURE_DEPENDS
  include/nanoeigenpy/*.hpp
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

set(${PROJECT_NAME}_SOURCES src/module.cpp src/solvers.cpp)
nanobind_add_module(nanoeigenpy NB_STATIC NB_SUPPRESS_WARNINGS ${nanoeigenpy_SOURCES} ${nanoeigenpy_HEADERS})
target_link_libraries(nanoeigenpy PRIVATE nanoeigenpy_headers)

# Cholmod
if(BUILD_WITH_CHOLMOD_SUPPORT)
  set(
    CMAKE_MODULE_PATH
    ${JRL_CMAKE_MODULES}/find-external/CHOLMOD
    ${CMAKE_MODULE_PATH}
  )
  ADD_PROJECT_DEPENDENCY(CHOLMOD REQUIRED FIND_EXTERNAL "CHOLMOD")
  message(
    STATUS
    "Build with CHOLMOD support (LGPL). See CHOLMOD/Doc/License.txt for further details."
  )
  file(
    GLOB ${PROJECT_NAME}_DECOMPOSITIONS_SPARSE_CHOLMOD_HEADERS
    include/nanoeigenpy/decompositions/sparse/cholmod/*.hpp
  )
  list(
    APPEND
    ${PROJECT_NAME}_HEADERS
    ${${PROJECT_NAME}_DECOMPOSITIONS_SPARSE_CHOLMOD_HEADERS}
  )
  target_link_libraries(nanoeigenpy PRIVATE CHOLMOD::CHOLMOD)
else()
  list(
    FILTER ${PROJECT_NAME}_HEADERS
    EXCLUDE
    REGEX "include/nanoeigenpy/decompositions/sparse/cholmod/.*"
  )
endif(BUILD_WITH_CHOLMOD_SUPPORT)

# Apple accelerate
if(BUILD_WITH_ACCELERATE_SUPPORT)
  if(NOT ${Eigen3_VERSION} VERSION_GREATER_EQUAL "3.4.90")
    message(
      FATAL_ERROR
      "Your version of Eigen is too low. Should be at least 3.4.90. Current version is ${Eigen3_VERSION}."
    )
  endif()

  set(
    CMAKE_MODULE_PATH
    ${JRL_CMAKE_MODULES}/find-external/Accelerate
    ${CMAKE_MODULE_PATH}
  )
  find_package(Accelerate REQUIRED)
  message(STATUS "Build with Accelerate support framework.")
  target_compile_definitions(
    nanoeigenpy_headers
    INTERFACE -DNANOEIGENPY_WITH_ACCELERATE_SUPPORT
  )
endif(BUILD_WITH_ACCELERATE_SUPPORT)

if(BUILD_WITH_ACCELERATE_SUPPORT)
  file(
    GLOB ${PROJECT_NAME}_DECOMPOSITIONS_SPARSE_ACCELERATE_HEADERS
    include/nanoeigenpy/decompositions/sparse/accelerate/*.hpp
  )
  list(
    APPEND
    ${PROJECT_NAME}_HEADERS
    ${${PROJECT_NAME}_DECOMPOSITIONS_SPARSE_ACCELERATE_HEADERS}
  )
else()
  list(
    FILTER ${PROJECT_NAME}_HEADERS
    EXCLUDE
    REGEX "include/nanoeigenpy/decompositions/sparse/accelerate/.*"
  )
endif(BUILD_WITH_ACCELERATE_SUPPORT)

if(BUILD_WITH_ACCELERATE_SUPPORT)
  target_link_libraries(nanoeigenpy PRIVATE Accelerate)
endif()

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

nanobind_add_stub(
  nanoeigenpy_stub
  INSTALL_TIME
  VERBOSE
  MODULE nanoeigenpy
  OUTPUT nanoeigenpy.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:nanoeigenpy>
  DEPENDS nanoeigenpy
)

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
  LIBRARY DESTINATION ${Python_SITELIB}
)

ADD_HEADER_GROUP(${PROJECT_NAME}_HEADERS)
ADD_SOURCE_GROUP(${PROJECT_NAME}_SOURCES)

SETUP_PROJECT_FINALIZE()
