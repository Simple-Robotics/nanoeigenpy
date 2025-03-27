/// Copyright 2025 INRIA

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "nanoeigenpy/decompositions.hpp"
#include "nanoeigenpy/geometry.hpp"
#include "nanoeigenpy/solvers.hpp"
#include "nanoeigenpy/utils/is-approx.hpp"
#include "nanoeigenpy/constants.hpp"

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
NB_MAKE_OPAQUE(Eigen::MINRES<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::HouseholderQR<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::FullPivHouseholderQR<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::ColPivHouseholderQR<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::EigenSolver<Eigen::MatrixXd>)

std::string printEigenVersion(const char* delim = ".") {
  std::ostringstream oss;
  oss << EIGEN_WORLD_VERSION << delim << EIGEN_MAJOR_VERSION << delim
      << EIGEN_MINOR_VERSION;
  return oss.str();
}

void exposeSolvers(nb::module_& m);

NB_MODULE(nanoeigenpy, m) {
  exposeConstants(m);

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

  exposeCholmodSimplicialLLT<SparseMatrix>(m, "CholmodSimplicialLLT");
  exposeCholmodSimplicialLDLT<SparseMatrix>(m, "CholmodSimplicialLDLT");
  exposeCholmodSupernodalLLT<SparseMatrix>(m, "CholmodSupernodalLLT");

  // Geometry
  exposeQuaternion<Scalar>(m, "Quaternion");
  exposeAngleAxis<Scalar>(m, "AngleAxis");

  // Preconditioners (and solvers)
  nb::module_ solvers = m.def_submodule("solvers", "Solvers in Eigen.");
  exposeSolvers(solvers);

  // Utils
  exposeIsApprox<double>(m);
  exposeIsApprox<std::complex<double>>(m);

  m.attr("__version__") = NANOEIGENPY_VERSION;
  m.attr("__eigen_version__") = printEigenVersion();

  m.def("SimdInstructionSetsInUse", &Eigen::SimdInstructionSetsInUse,
        "Get the set of SIMD instructions used in Eigen when this module was "
        "compiled.");
}

void exposeSolvers(nb::module_& m) {
  exposeIdentityPreconditioner<Scalar>(m, "IdentityPreconditioner");
  exposeDiagonalPreconditioner<Scalar>(m, "DiagonalPreconditioner");
#if EIGEN_VERSION_AT_LEAST(3, 3, 5)
  exposeLeastSquareDiagonalPreconditioner<Scalar>(
      m, "LeastSquareDiagonalPreconditioner");
#endif

  // Solvers
  using Eigen::Lower;
  using Eigen::Upper;
  using ConjugateGradient = Eigen::ConjugateGradient<Matrix, Lower | Upper>;
  exposeConjugateGradient<ConjugateGradient>(m, "ConjugateGradient");
  using LeastSquaresConjugateGradient = Eigen::LeastSquaresConjugateGradient<
      Matrix, Eigen::LeastSquareDiagonalPreconditioner<Scalar>>;
  exposeLeastSquaresConjugateGradient<LeastSquaresConjugateGradient>(
      m, "LeastSquaresConjugateGradient");

  using IdentityConjugateGradient =
      Eigen::ConjugateGradient<Matrix, Lower | Upper,
                               Eigen::IdentityPreconditioner>;
  exposeConjugateGradient<IdentityConjugateGradient>(
      m, "IdentityConjugateGradient");
}
