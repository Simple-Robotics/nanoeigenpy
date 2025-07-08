/// Copyright 2025 INRIA

#include <nanobind/stl/string.h>

#include "nanoeigenpy/decompositions.hpp"
#include "nanoeigenpy/decompositions/generalized-eigen-solver.hpp"
#include "nanoeigenpy/geometry.hpp"
#include "nanoeigenpy/utils/is-approx.hpp"
#include "nanoeigenpy/constants.hpp"

#include "./internal.h"

using namespace nanoeigenpy;

using Quaternion = Eigen::Quaternion<Scalar, Options>;
using SparseMatrix = Eigen::SparseMatrix<Scalar, Options>;

NB_MAKE_OPAQUE(Eigen::LLT<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::LDLT<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::HouseholderQR<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::FullPivHouseholderQR<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::ColPivHouseholderQR<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::EigenSolver<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::ComplexEigenSolver<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::ComplexSchur<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::GeneralizedEigenSolver<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::HessenbergDecomposition<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::RealQZ<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::RealSchur<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::Tridiagonalization<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::FullPivLU<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::PartialPivLU<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::BDCSVD<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::JacobiSVD<Eigen::MatrixXd>)

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
  exposeHouseholderQRSolver<Matrix>(m, "HouseholderQR");
  exposeFullPivHouseholderQRSolver<Matrix>(m, "FullPivHouseholderQR");
  exposeColPivHouseholderQRSolver<Matrix>(m, "ColPivHouseholderQR");
  exposeCompleteOrthogonalDecompositionSolver<Matrix>(
      m, "CompleteOrthogonalDecomposition");
  exposeEigenSolver<Matrix>(m, "EigenSolver");
  exposeSelfAdjointEigenSolver<Matrix>(m, "SelfAdjointEigenSolver");
  exposeGeneralizedSelfAdjointEigenSolver<Matrix>(
      m, "GeneralizedSelfAdjointEigenSolver");
  exposeComplexEigenSolver<Matrix>(m, "ComplexEigenSolver");
  exposeComplexSchur<Matrix>(m, "ComplexSchur");
  exposeGeneralizedEigenSolver<Matrix>(m, "GeneralizedEigenSolver");
  exposeHessenbergDecomposition<Matrix>(m, "HessenbergDecomposition");
  exposeRealQZ<Matrix>(m, "RealQZ");
  exposeRealSchur<Matrix>(m, "RealSchur");
  exposeTridiagonalization<Matrix>(m, "Tridiagonalization");
  exposePermutationMatrix<Eigen::Dynamic>(m, "PermutationMatrix");
  exposeFullPivLUSolver<Matrix>(m, "FullPivLU");
  exposePartialPivLUSolver<Matrix>(m, "PartialPivLU");
  exposeBDCSVDSolver<Matrix>(m, "BDCSVD");
  exposeJacobiSVDSolver<Matrix>(m, "JacobiSVD");

  exposeSimplicialLLT<SparseMatrix>(m, "SimplicialLLT");
  exposeSimplicialLDLT<SparseMatrix>(m, "SimplicialLDLT");
  exposeSparseLU<SparseMatrix>(m, "SparseLU");
  exposeSparseQR<SparseMatrix>(m, "SparseQR");

#ifdef NANOEIGENPY_HAS_CHOLMOD
  exposeCholmodSimplicialLLT<SparseMatrix>(m, "CholmodSimplicialLLT");
  exposeCholmodSimplicialLDLT<SparseMatrix>(m, "CholmodSimplicialLDLT");
  exposeCholmodSupernodalLLT<SparseMatrix>(m, "CholmodSupernodalLLT");
#endif
#ifdef NANOEIGENPY_HAS_ACCELERATE
  exposeAccelerate(m);
#endif

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
