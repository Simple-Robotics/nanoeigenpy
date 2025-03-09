/// Copyright 2025 INRIA
#pragma once
#include "base.hpp"
#include <Eigen/Cholesky>

namespace nanoeigenpy {
namespace nb = nanobind;

template <typename MatrixType, typename MatrixOrVector>
MatrixOrVector solve(const Eigen::LLT<MatrixType> &c, const MatrixOrVector &vec) {
     return c.solve(vec);
}

template <typename MatrixType>
void exposeLLTSolver(nb::module_ m, const char *name) {
  using Chol = Eigen::LLT<MatrixType>;
  using Scalar = typename MatrixType::Scalar;
  using VectorType = Eigen::Matrix<Scalar, -1, 1>;
  auto cl = nb::class_<Chol>(m, name, 
                "Standard Cholesky decomposition (LL^T) of a matrix and associated "
                "features.\n\n"
                "This class performs a LL^T Cholesky decomposition of a symmetric, "
                "positive definite matrix A such that A = LL^* = U^*U, where L is "
                "lower triangular.\n\n"
                "While the Cholesky decomposition is particularly useful to solve "
                "selfadjoint problems like D^*D x = b, for that purpose, we recommend "
                "the Cholesky decomposition without square root which is more stable "
                "and even faster. Nevertheless, this standard Cholesky decomposition "
                "remains useful in many other situations like generalised eigen "
                "problems with hermitian matrices.")

                .def(nb::init<>(),                                                              
                "Default constructor.")
                .def(nb::init<Eigen::DenseIndex>(), nb::arg("size"),                       
                "Default constructor with memory preallocation.")
                .def(nb::init<const MatrixType &>(), nb::arg("matrix"),                   
                "Constructs a LLT factorization from a given matrix.")

                .def(EigenBaseVisitor())                                                         

                .def("matrixL",                                                                 
                     [](Chol const &c) -> MatrixType { return c.matrixL(); },
                     "Returns the lower triangular matrix L.")
                .def("matrixU",                                                                 
                     [](Chol const &c) -> MatrixType { return c.matrixU(); },
                     "Returns the upper triangular matrix U.")
                .def("matrixLLT", &Chol::matrixLLT,                                             
                     "Returns the LLT decomposition matrix.",
                     nb::rv_policy::reference_internal)

#if EIGEN_VERSION_AT_LEAST(3, 3, 90)               
                .def("rankUpdate",                                                               
                    [](Chol &c, VectorType const &w, Scalar sigma) {
                      return c.rankUpdate(w, sigma);
                    },
                    nb::arg("w"), nb::arg("sigma"),                          
                    nb::rv_policy::reference)
#else
                .def("rankUpdate",
                    [](Chol &c, VectorType const &w, Scalar sigma) {
                      return c.rankUpdate(w, sigma);
                    },
                    nb::arg("w"), nb::arg("sigma"))
#endif

#if EIGEN_VERSION_AT_LEAST(3, 3, 0)
                .def("adjoint", &Chol::adjoint,                                                  
                     "Returns the adjoint, that is, a reference to the decomposition "
                     "itself as if the underlying matrix is self-adjoint.",
                     nb::rv_policy::reference)
#endif

                .def("compute",                                                                 
                    [](Chol &c, VectorType const &matrix) {
                      return c.compute(matrix);
                    },
                    nb::arg("matrix"),
                    "Computes the LDLT of given matrix.",
                    nb::rv_policy::reference)
                .def("info", &Chol::info,                                                       
                     "NumericalIssue if the input contains INF or NaN values or "
                     "overflow occured. Returns Success otherwise.")

#if EIGEN_VERSION_AT_LEAST(3, 3, 0)
                .def("rcond", &Chol::rcond,                                                       
                     "Returns an estimate of the reciprocal condition number of the "
                     "matrix.")
#endif

                .def("reconstructedMatrix", &Chol::reconstructedMatrix,                         
                     "Returns the matrix represented by the decomposition, i.e., it "
                     "returns the product: L L^*. This function is provided for debug "
                     "purpose.")

                .def("solve",                                                                        
                     [](Chol const &c, VectorType const &b) -> VectorType { return solve(c, b); },
                     nb::arg("b"),
                     "Returns the solution x of A x = b using the current "
                     "decomposition of A.")
                .def("solve",                                                                        
                     [](Chol const &c, MatrixType const &B) -> MatrixType { return solve(c, B); },
                     nb::arg("B"),
                     "Returns the solution X of A X = B using the current "
                     "decomposition of A where B is a right hand side matrix.")

                .def("id",                                                                             
                     [](Chol const &c) -> int64_t { return reinterpret_cast<int64_t>(&c); },
                     "Returns the unique identity of an object.\n"
                     "For objects held in C++, it corresponds to its memory address.");

}

}  // namespace nanoeigenpy


// TODO

// Tests that were not done in eigenpy that we could add in nanoeigenpy:
// Default constructor
// Default constructor with memory preallocation
// matrixLLT
// rankUpdate
// adjoint
// compute
// rcond
// reconstructedMatrix

// Expose supplementary content:
// Expose ComputationInfo to test info

// Assertions for the solve method on vectors x_est and b
// Expose is_approx for vectors too (exposed and tested for ùatrices only in eigenpy)
