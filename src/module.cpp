/// Copyright 2025 INRIA
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "nanoeigenpy/decompositions.hpp"
#include "nanoeigenpy/geometry.hpp"
#include "nanoeigenpy/utils/is-approx.hpp"

using namespace nanoeigenpy;

using Scalar = double;
static constexpr int Options = Eigen::ColMajor;
using Matrix = Eigen::Matrix<Scalar, -1, -1, Options>;
using Vector = Eigen::Matrix<Scalar, -1, 1>;
using Quaternion = Eigen::Quaternion<Scalar, Options>;

NB_MAKE_OPAQUE(Eigen::LLT<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::LDLT<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::HouseholderQR<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::FullPivHouseholderQR<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::ColPivHouseholderQR<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd>)

NB_MODULE(nanoeigenpy, m) {
  // Decompositions
  exposeLLTSolver<Matrix>(m, "LLT");
  exposeLDLTSolver<Matrix>(m, "LDLT");
  exposeMINRESSolver<Matrix>(m, "MINRES");
  exposeHouseholderQRSolver<Matrix>(m, "HouseholderQR");
  exposeFullPivHouseholderQRSolver<Matrix>(m, "FullPivHouseholderQR");
  exposeColPivHouseholderQRSolver<Matrix>(m, "ColPivHouseholderQR");
  exposeCompleteOrthogonalDecompositionSolver<Matrix>(m, "CompleteOrthogonalDecomposition");
  exposeEigenSolver<Matrix>(m, "exposeEigenSolver");

  // Geometry
  exposeQuaternion<Scalar>(m, "Quaternion");
  exposeAngleAxis<double>(m, "AngleAxis");

  // Utils
  exposeIsApprox<double>(m);
  exposeIsApprox<std::complex<double>>(m);
}


// Meeting Mar 11 - About nanoeigenpy

// In eigenpy, there are decompositions, solvers, geometry stuff, etc
// -> I focused on some decompositions first: LLT, LDLT, MINRES, QR, EigenSolver

// What I did:
// 1. Translated the Bosst.Python classes of visitors into nanobind classes in expose functions
// 2. Passed the same tests as those in eigenpy

// TODO: 
// 1. Finish to expose decompositions and geometry stuff
// 2. Upgrade the unit tests to all the methods (defined in .def() so that they are more complete)
// 3. Additional content from eigenpy (eg the diferent solvers, etc)
// (2 <-> 3 ?)

