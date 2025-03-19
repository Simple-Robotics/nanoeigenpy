/// Copyright 2025 INRIA
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "nanoeigenpy/decompositions.hpp"
#include "nanoeigenpy/decompositions/sparse/ldlt.hpp"
#include "nanoeigenpy/geometry.hpp"
#include "nanoeigenpy/utils/is-approx.hpp"
#include "nanoeigenpy/computation-info.hpp"

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
  exposeCompleteOrthogonalDecompositionSolver<Matrix>(
      m, "CompleteOrthogonalDecomposition");
  exposeEigenSolver<Matrix>(m, "EigenSolver");

  exposeSimplicialLLT<Matrix>(m, "SimplicialLLT");
  exposeSimplicialLDLT<Matrix>(m, "SimplicialLDLT");

  // Geometry
  // exposeQuaternion<Scalar>(m, "Quaternion");
  // exposeAngleAxis<double>(m, "AngleAxis");

  // Utils
  exposeIsApprox<double>(m);
  exposeIsApprox<std::complex<double>>(m);

  exposeComputationInfo(m);
}
