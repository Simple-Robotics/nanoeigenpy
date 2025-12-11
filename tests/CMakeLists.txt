nanobind_add_module(quaternion
  NB_STATIC LTO NB_SUPPRESS_WARNINGS
  quaternion.cpp
)
target_link_libraries(quaternion PRIVATE Eigen3::Eigen)
jrl_target_set_output_directory(quaternion OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test-quaternion)

# gersemi: off
function(nanoeigenpy_add_test name)
  set(test_name "nanoeigenpy: ${name}")
  add_test(
    NAME ${test_name}
    COMMAND $<TARGET_FILE:Pytest::Pytest> ${CMAKE_CURRENT_SOURCE_DIR}/test_${name}.py
  )
  set_property(
    TEST ${test_name}
    PROPERTY ENVIRONMENT_MODIFICATION
        "PYTHONPATH=path_list_prepend:$<TARGET_FILE_DIR:nanoeigenpy>"
        "PYTHONPATH=path_list_prepend:$<TARGET_FILE_DIR:quaternion>"
  )
endfunction()
# gersemi: on

nanoeigenpy_add_test(bdcsvd)
nanoeigenpy_add_test(complex_eigen_solver)
nanoeigenpy_add_test(complex_schur)
nanoeigenpy_add_test(eigen_solver)
nanoeigenpy_add_test(full_piv_lu)
nanoeigenpy_add_test(generalized_eigen_solver)
nanoeigenpy_add_test(generalized_self_adjoint_eigen_solver)
nanoeigenpy_add_test(geometry)
nanoeigenpy_add_test(hessenberg_decomposition)
nanoeigenpy_add_test(import_extension)
nanoeigenpy_add_test(incomplete_cholesky)
nanoeigenpy_add_test(incomplete_lut)
nanoeigenpy_add_test(iterative_solvers)
nanoeigenpy_add_test(jacobi_svd)
nanoeigenpy_add_test(ldlt)
nanoeigenpy_add_test(llt)
nanoeigenpy_add_test(partial_piv_lu)
nanoeigenpy_add_test(permutation_matrix)
nanoeigenpy_add_test(qr)
nanoeigenpy_add_test(real_qz)
nanoeigenpy_add_test(real_schur)
nanoeigenpy_add_test(self_adjoint_eigen_solver)
nanoeigenpy_add_test(simplicial_llt)
nanoeigenpy_add_test(sparse_lu)
nanoeigenpy_add_test(sparse_qr)
nanoeigenpy_add_test(tridiagonalization)

if(BUILD_WITH_CHOLMOD_SUPPORT)
  nanoeigenpy_add_test(cholmod_simplicial_ldlt)
  nanoeigenpy_add_test(cholmod_simplicial_llt)
  nanoeigenpy_add_test(cholmod_supernodal_llt)
endif()

if(BUILD_WITH_ACCELERATE_SUPPORT)
  nanoeigenpy_add_test(accelerate)
endif()
