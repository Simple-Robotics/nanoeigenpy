# Create a shared nanobind library for testing
set(NANOBIND_TESTING_TARGET nanobind-testing)
nanobind_build_library(${NANOBIND_TESTING_TARGET} SHARED)

# On Win32, shared DLL libs are sent to RUNTIME_OUTPUT_DIRECTORY, *but*
# we really need to send it to the lib dir so
if(WIN32)
  set_target_properties(
    ${NANOBIND_TESTING_TARGET}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
  )
endif()

# Add a C++ extension module for tests
function(add_tests_cpp_extension name)
  set(filename ${name}.cpp)
  add_library(${name} MODULE ${filename})
  target_link_libraries(
    ${name}
    PRIVATE ${NANOBIND_TESTING_TARGET} nanoeigenpy_headers
  )
  # Use nanobind low-level interface to set properties
  nanobind_set_visibility(${name})
  nanobind_strip(${name})
  nanobind_extension(${name})
  nanobind_compile_options(${name})
  nanobind_link_options(${name})

  add_dependencies(build_tests ${name})

  add_test(
    NAME "${PROJECT_NAME}-import-${name}"
    COMMAND ${Python_EXECUTABLE} -c "import ${name}"
    WORKING_DIRECTORY $<TARGET_FILE_DIR:${name}>
  )
endfunction()

# Add Python test module
function(add_tests_py_module name)
  set(filename tests/${name}.py)
  set(test_target "${PROJECT_NAME}-${name}")
  string(REPLACE "_" "-" test_target ${test_target})
  set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
  ADD_PYTHON_UNIT_TEST(${test_target} ${filename} "lib")
  unset(PYTHON_EXECUTABLE)
  set_tests_properties(${test_target} PROPERTIES DEPENDS nanoeigenpy)
endfunction()

add_dependencies(build_tests nanoeigenpy)

add_test(
  NAME "${PROJECT_NAME}-import-extension"
  COMMAND ${Python_EXECUTABLE} -c "import ${PROJECT_NAME}"
  WORKING_DIRECTORY $<TARGET_FILE_DIR:${PROJECT_NAME}>
)

add_tests_cpp_extension(quaternion)

set(
  TEST_NAMES
  test_eigen_solver
  test_complex_eigen_solver
  test_generalized_eigen_solver
  test_self_adjoint_eigen_solver
  test_generalized_self_adjoint_eigen_solver
  test_real_schur
  test_complex_schur
  test_hessenberg_decomposition
  test_real_qz
  test_tridiagonalization
  test_bdcsvd
  test_jacobi_svd
  test_full_piv_lu
  test_partial_piv_lu
  test_ldlt
  test_llt
  test_qr
  test_simplicial_llt
  test_sparse_lu
  test_sparse_qr
  test_geometry
  test_iterative_solvers
  test_permutation_matrix
)

if(BUILD_WITH_CHOLMOD_SUPPORT)
  list(
    APPEND
    TEST_NAMES
    test_cholmod_simplicial_ldlt
    test_cholmod_simplicial_llt
    test_cholmod_supernodal_llt
  )
endif(BUILD_WITH_CHOLMOD_SUPPORT)

foreach(test_name ${TEST_NAMES})
  message(STATUS "Adding Python test ${test_name}")
  add_tests_py_module(${test_name})
endforeach()

if(BUILD_WITH_ACCELERATE_SUPPORT)
  message(STATUS "Adding Python test test_accelerate")
  add_tests_py_module(test_accelerate)
endif(BUILD_WITH_ACCELERATE_SUPPORT)
