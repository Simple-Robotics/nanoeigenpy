# Copyright 2025-2026 Inria

cmake_minimum_required(VERSION 3.22...4.2)

project(
  nanoeigenpy
  VERSION 0.12.0
  DESCRIPTION "A support library for bindings between Eigen in C++ and Python, based on nanobind"
  HOMEPAGE_URL "https://github.com/Simple-Robotics/nanoeigenpy"
)

include(cmake/get-jrl-cmakemodules.cmake)

jrl_configure_defaults()

jrl_option(BUILD_TESTING "Build the tests" OFF)

jrl_option(INSTALL_DOCUMENTATION "Generate and install the documentation" OFF)

jrl_option(BUILD_WITH_CHOLMOD_SUPPORT
      "Build EigenPy with the Cholmod (LGPL) support. See CHOLMOD/Doc/License.txt for further details." OFF
)

jrl_option(BUILD_WITH_ACCELERATE_SUPPORT
        "Build EigenPy with the Accelerate support (Apple only)" OFF
)

if(BUILD_WITH_ACCELERATE_SUPPORT AND NOT APPLE)
  message(WARNING "Accelerate support is only available on APPLE systems")
endif()

jrl_find_package(Eigen3 CONFIG REQUIRED)

jrl_find_python(3.8 REQUIRED COMPONENTS Interpreter Development.Module)
jrl_find_nanobind(2.5.0 CONFIG QUIET)

if(nanobind_ROOT AND NOT nanobind_FOUND)
  message(WARNING "nanobind found at ${nanobind_ROOT} but too old")
endif()

# On Ubuntu 24.04, the nanobind-dev package ships nanobind 1.9.2.
# We require >=2.5.0 for nanobind_add_stub,
# NB_SUPPRESS_WARNINGS, and visitor pattern (c++)
if(NOT nanobind_FOUND)
  set(nanobind_GIT_REPOSITORY "https://github.com/wjakob/nanobind.git")
  set(nanobind_GIT_TAG "v2.10.1")

  message(
    STATUS
    "nanobind: fallback to FetchContent from ${nanobind_GIT_REPOSITORY} (tag: ${nanobind_GIT_TAG})"
  )

  include(FetchContent)
  FetchContent_Declare(
    nanobind
    GIT_REPOSITORY ${nanobind_GIT_REPOSITORY}
    GIT_TAG ${nanobind_GIT_TAG}
    GIT_SHALLOW True
  )
  FetchContent_MakeAvailable(nanobind)
endif()

if(BUILD_WITH_CHOLMOD_SUPPORT)
  jrl_find_package(CHOLMOD CONFIG REQUIRED)
endif()

if(BUILD_WITH_ACCELERATE_SUPPORT AND APPLE)
  jrl_find_package(Accelerate REQUIRED)
endif()

if(BUILD_TESTING)
  jrl_find_package(Pytest REQUIRED)
endif()

set(
  nanoeigenpy_HEADERS
  include/nanoeigenpy/decompositions/sparse/simplicial-cholesky.hpp
  include/nanoeigenpy/decompositions/sparse/simplicial-ldlt.hpp
  include/nanoeigenpy/decompositions/sparse/simplicial-llt.hpp
  include/nanoeigenpy/decompositions/sparse/sparse-lu.hpp
  include/nanoeigenpy/decompositions/sparse/sparse-qr.hpp
  include/nanoeigenpy/decompositions/sparse/sparse-solver-base.hpp
  include/nanoeigenpy/decompositions/bdcsvd.hpp
  include/nanoeigenpy/decompositions/col-piv-householder-qr.hpp
  include/nanoeigenpy/decompositions/complete-orthogonal-decomposition.hpp
  include/nanoeigenpy/decompositions/complex-eigen-solver.hpp
  include/nanoeigenpy/decompositions/complex-schur.hpp
  include/nanoeigenpy/decompositions/eigen-solver.hpp
  include/nanoeigenpy/decompositions/full-piv-householder-qr.hpp
  include/nanoeigenpy/decompositions/full-piv-lu.hpp
  include/nanoeigenpy/decompositions/generalized-eigen-solver.hpp
  include/nanoeigenpy/decompositions/generalized-self-adjoint-eigen-solver.hpp
  include/nanoeigenpy/decompositions/hessenberg-decomposition.hpp
  include/nanoeigenpy/decompositions/householder-qr.hpp
  include/nanoeigenpy/decompositions/jacobi-svd.hpp
  include/nanoeigenpy/decompositions/ldlt.hpp
  include/nanoeigenpy/decompositions/llt.hpp
  include/nanoeigenpy/decompositions/partial-piv-lu.hpp
  include/nanoeigenpy/decompositions/permutation-matrix.hpp
  include/nanoeigenpy/decompositions/real-qz.hpp
  include/nanoeigenpy/decompositions/real-schur.hpp
  include/nanoeigenpy/decompositions/self-adjoint-eigen-solver.hpp
  include/nanoeigenpy/decompositions/svd-base.hpp
  include/nanoeigenpy/decompositions/tridiagonalization.hpp
  include/nanoeigenpy/geometry/detail/rotation-base.hpp
  include/nanoeigenpy/geometry/angle-axis.hpp
  include/nanoeigenpy/geometry/hyperplane.hpp
  include/nanoeigenpy/geometry/jacobi-rotation.hpp
  include/nanoeigenpy/geometry/parametrized-line.hpp
  include/nanoeigenpy/geometry/quaternion.hpp
  include/nanoeigenpy/geometry/rotation-2d.hpp
  include/nanoeigenpy/geometry/scaling.hpp
  include/nanoeigenpy/geometry/translation.hpp
  include/nanoeigenpy/solvers/basic-preconditioners.hpp
  include/nanoeigenpy/solvers/bfgs-preconditioners.hpp
  include/nanoeigenpy/solvers/bicgstab.hpp
  include/nanoeigenpy/solvers/conjugate-gradient.hpp
  include/nanoeigenpy/solvers/incomplete-cholesky.hpp
  include/nanoeigenpy/solvers/incomplete-lut.hpp
  include/nanoeigenpy/solvers/iterative-solver-base.hpp
  include/nanoeigenpy/solvers/least-squares-conjugate-gradient.hpp
  include/nanoeigenpy/solvers/minres.hpp
  include/nanoeigenpy/utils/helpers.hpp
  include/nanoeigenpy/utils/is-approx.hpp
  include/nanoeigenpy/constants.hpp
  include/nanoeigenpy/decompositions.hpp
  include/nanoeigenpy/eigen-base.hpp
  include/nanoeigenpy/fwd.hpp
  include/nanoeigenpy/geometry.hpp
  include/nanoeigenpy/id.hpp
  include/nanoeigenpy/nanoeigenpy.hpp
  include/nanoeigenpy/solvers.hpp
)

if(BUILD_WITH_CHOLMOD_SUPPORT)
  list(
    APPEND
    nanoeigenpy_HEADERS
    include/nanoeigenpy/decompositions/sparse/cholmod/cholmod-base.hpp
    include/nanoeigenpy/decompositions/sparse/cholmod/cholmod-decomposition.hpp
    include/nanoeigenpy/decompositions/sparse/cholmod/cholmod-simplicial-ldlt.hpp
    include/nanoeigenpy/decompositions/sparse/cholmod/cholmod-simplicial-llt.hpp
    include/nanoeigenpy/decompositions/sparse/cholmod/cholmod-supernodal-llt.hpp
  )
endif()

if(BUILD_WITH_ACCELERATE_SUPPORT)
  list(
    APPEND
    nanoeigenpy_HEADERS
    include/nanoeigenpy/decompositions/sparse/accelerate/accelerate.hpp
  )
endif()

# ----------------------------------------------------------------------- #

add_library(nanoeigenpy_headers INTERFACE)
add_library(nanoeigenpy::nanoeigenpy_headers ALIAS nanoeigenpy_headers)
target_compile_features(nanoeigenpy_headers INTERFACE cxx_std_17)

target_include_directories(
  nanoeigenpy_headers
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

set(nanoeigenpy_SOURCES src/module.cpp)

nanobind_add_module(nanoeigenpy
  NB_STATIC LTO NB_SUPPRESS_WARNINGS
  ${nanoeigenpy_SOURCES}
  ${nanoeigenpy_HEADERS}
)
jrl_target_set_output_directory(nanoeigenpy OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/site-packages)

jrl_target_enforce_msvc_conformance(nanoeigenpy PRIVATE)
jrl_target_generate_config_header(nanoeigenpy PRIVATE VERSION ${PROJECT_VERSION})
jrl_target_generate_warning_header(nanoeigenpy PRIVATE)
jrl_target_generate_deprecated_header(nanoeigenpy PRIVATE)
jrl_check_python_module_name(nanoeigenpy)

target_link_libraries(nanoeigenpy PRIVATE nanoeigenpy_headers Eigen3::Eigen)

if(BUILD_WITH_CHOLMOD_SUPPORT)
  target_link_libraries(nanoeigenpy PRIVATE SuiteSparse::CHOLMOD)
  target_compile_definitions(nanoeigenpy PRIVATE NANOEIGENPY_HAS_CHOLMOD)
endif()

if(BUILD_WITH_ACCELERATE_SUPPORT AND APPLE)
  target_link_libraries(nanoeigenpy PRIVATE Accelerate::Accelerate)
  target_compile_definitions(nanoeigenpy PRIVATE NANOEIGENPY_HAS_ACCELERATE)
endif()

# Stub generation requires typing-extensions
# ROS Humble ships an incompatible typing-extensions with python3.10
if(Python_VERSION VERSION_GREATER_EQUAL 3.11.0)
  nanobind_add_stub(
    nanoeigenpy_stub
    VERBOSE
    MODULE nanoeigenpy
    OUTPUT ${CMAKE_BINARY_DIR}/lib/site-packages/nanoeigenpy.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:nanoeigenpy>
    DEPENDS nanoeigenpy
  )
endif()

jrl_python_compute_install_dir(python_install_dir)
# NOTE: install the whole binary dir, we need the "/" at the end to copy the content.
# In a next version, we might install in a nanoeigenpy directory, which contains all the files.
install(
  DIRECTORY ${CMAKE_BINARY_DIR}/lib/site-packages/
  DESTINATION ${python_install_dir}
  FILES_MATCHING
  PATTERN "*.py"
  PATTERN "*.pyc"
  PATTERN "*.pyi"
  PATTERN "*.typed"
  PATTERN "*.so"
  PATTERN "*.pyd"
)
# ------------------------------------------------------------------------ #
jrl_target_headers(nanoeigenpy_headers INTERFACE
  HEADERS ${nanoeigenpy_HEADERS}
  BASE_DIRS include
)

jrl_add_export_component(NAME nanoeigenpy_headers TARGETS nanoeigenpy_headers)
jrl_export_package()

# ------------------------------------------------------------------------ #
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

jrl_print_dependencies_summary()
jrl_print_options_summary()
