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
// TODO: Same for geometry stuff ?

NB_MODULE(nanoeigenpy, m) {
  // Decompositions
  exposeLLTSolver<Matrix>(m, "LLT");
  exposeLDLTSolver<Matrix>(m, "LDLT");
  exposeMINRESSolver<Matrix>(m, "MINRES");
  // exposeHouseholderQRSolver<Matrix>(m, "HouseholderQR");
  // exposeFullPivHouseholderQRSolver<Matrix>(m, "FullPivHouseholderQR")
  // exposeColPivHouseholderQRSolver<Matrix>(m, "ColPivHouseholderQR")
  // exposeCompleteOrthogonalDecompositionSolver<Matrix>(m, "CompleteOrthogonalDecomposition")

  // Geometry
  exposeQuaternion<Scalar>(m, "Quaternion");
  exposeAngleAxis<double>(m, "AngleAxis");

  // Utils
  exposeIsApprox<double>(m);
  exposeIsApprox<std::complex<double>>(m);
}
