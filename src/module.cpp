/// Copyright 2025 INRIA

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "nanoeigenpy/decompositions.hpp"
#include "nanoeigenpy/geometry.hpp"
#include "nanoeigenpy/solvers.hpp"
#include "nanoeigenpy/utils/is-approx.hpp"
#include "nanoeigenpy/computation-info.hpp"

using namespace nanoeigenpy;
namespace nb = nanobind;

using Scalar = double;
static constexpr int Options = Eigen::ColMajor;
using Matrix = Eigen::Matrix<Scalar, -1, -1, Options>;
using Vector = Eigen::Matrix<Scalar, -1, 1>;
using Quaternion = Eigen::Quaternion<Scalar, Options>;
using SparseMatrix = Eigen::SparseMatrix<Scalar, Options>;

NB_MAKE_OPAQUE(Eigen::LLT<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::LDLT<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::MINRES<Eigen::MatrixXd>)  // necessary ?
NB_MAKE_OPAQUE(Eigen::HouseholderQR<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::FullPivHouseholderQR<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::ColPivHouseholderQR<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::EigenSolver<Eigen::MatrixXd>)  // necessary ?

std::string printEigenVersion(const char* delim = ".") {
  std::ostringstream oss;
  oss << EIGEN_WORLD_VERSION << delim << EIGEN_MAJOR_VERSION << delim
      << EIGEN_MINOR_VERSION;
  return oss.str();
}

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
  exposeSelfAdjointEigenSolver<Matrix>(m, "SelfAdjointEigenSolver");
  exposePermutationMatrix<Eigen::Dynamic>(m, "PermutationMatrix");

  exposeSimplicialLLT<SparseMatrix>(m, "SimplicialLLT");
  exposeSimplicialLDLT<SparseMatrix>(m, "SimplicialLDLT");

  // Geometry
  exposeQuaternion<Scalar>(m, "Quaternion");
  exposeAngleAxis<Scalar>(m, "AngleAxis");

  // Preconditioners (and solvers)
  nb::module_ solvers = m.def_submodule("solvers", "Solvers in Eigen.");

  exposeIdentityPreconditioner<Scalar>(solvers, "IdentityPreconditioner");
  exposeDiagonalPreconditioner<Scalar>(solvers, "DiagonalPreconditioner");
#if EIGEN_VERSION_AT_LEAST(3, 3, 5)
  exposeLeastSquareDiagonalPreconditioner<Scalar>(
      solvers, "LeastSquareDiagonalPreconditioner");
#endif

  // // Solvers
  using namespace Eigen;
  using ConjugateGradient = ConjugateGradient<MatrixXd, Lower | Upper>;
  exposeConjugateGradient<ConjugateGradient>(solvers, "ConjugateGradient");
  using LeastSquaresConjugateGradient = LeastSquaresConjugateGradient<
      MatrixXd, LeastSquareDiagonalPreconditioner<MatrixXd::Scalar>>;
  exposeLeastSquaresConjugateGradient<LeastSquaresConjugateGradient>(
      solvers, "LeastSquaresConjugateGradient");

  using IdentityConjugateGradient =
      Eigen::ConjugateGradient<MatrixXd, Lower | Upper, IdentityPreconditioner>;
  exposeConjugateGradient<IdentityConjugateGradient>(
      solvers, "IdentityConjugateGradient");

  // Utils
  exposeIsApprox<double>(m);
  exposeIsApprox<std::complex<double>>(m);

  exposeComputationInfo(m);

  m.attr("__version__") = NANOEIGENPY_VERSION;
  m.attr("__eigen_version__") = printEigenVersion();

  m.def("SimdInstructionSetsInUse", &Eigen::SimdInstructionSetsInUse,
        "Get the set of SIMD instructions used in Eigen when this module was "
        "compiled.");
}
