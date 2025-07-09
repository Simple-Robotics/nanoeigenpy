/// Copyright 2025 INRIA

#include <nanobind/stl/string.h>

#include "nanoeigenpy/decompositions.hpp"
#include "nanoeigenpy/geometry.hpp"
#include "nanoeigenpy/utils/is-approx.hpp"
#include "nanoeigenpy/constants.hpp"

#include "./internal.h"

using namespace nanoeigenpy;

using Quaternion = Eigen::Quaternion<Scalar, Options>;
using SparseMatrix = Eigen::SparseMatrix<Scalar, Options>;

NB_MAKE_OPAQUE(Eigen::LLT<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::LDLT<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::FullPivLU<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::PartialPivLU<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::ColPivHouseholderQR<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::FullPivHouseholderQR<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::HouseholderQR<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::BDCSVD<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::JacobiSVD<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::ComplexEigenSolver<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::ComplexSchur<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::EigenSolver<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::GeneralizedEigenSolver<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::HessenbergDecomposition<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::RealQZ<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::RealSchur<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::Tridiagonalization<Eigen::MatrixXd>)

std::string printEigenVersion(const char* delim = ".") {
  std::ostringstream oss;
  oss << EIGEN_WORLD_VERSION << delim << EIGEN_MAJOR_VERSION << delim
      << EIGEN_MINOR_VERSION;
  return oss.str();
}

void exposeSolvers(nb::module_& m);

NB_MODULE(nanoeigenpy, m) {
  // <Eigen/Core>
  exposeConstants(m);
  exposePermutationMatrix<Eigen::Dynamic>(m, "PermutationMatrix");

  // <Eigen/Cholesky>
  exposeLDLT<Matrix>(m, "LDLT");
  exposeLLT<Matrix>(m, "LLT");
  // <Eigen/LU>
  exposeFullPivLU<Matrix>(m, "FullPivLU");
  exposePartialPivLU<Matrix>(m, "PartialPivLU");
  // <Eigen/QR>
  exposeColPivHouseholderQR<Matrix>(m, "ColPivHouseholderQR");
  exposeCompleteOrthogonalDecomposition<Matrix>(
      m, "CompleteOrthogonalDecomposition");
  exposeFullPivHouseholderQR<Matrix>(m, "FullPivHouseholderQR");
  exposeHouseholderQR<Matrix>(m, "HouseholderQR");
  // <Eigen/SVD>
  exposeBDCSVD<Matrix>(m, "BDCSVD");
  exposeJacobiSVD<Matrix>(m, "JacobiSVD");
  // <Eigen/Eigenvalues>
  exposeComplexEigenSolver<Matrix>(m, "ComplexEigenSolver");
  exposeComplexSchur<Matrix>(m, "ComplexSchur");
  exposeEigenSolver<Matrix>(m, "EigenSolver");
  exposeGeneralizedEigenSolver<Matrix>(m, "GeneralizedEigenSolver");
  exposeGeneralizedSelfAdjointEigenSolver<Matrix>(
      m, "GeneralizedSelfAdjointEigenSolver");
  exposeHessenbergDecomposition<Matrix>(m, "HessenbergDecomposition");
  exposeRealQZ<Matrix>(m, "RealQZ");
  exposeRealSchur<Matrix>(m, "RealSchur");
  exposeSelfAdjointEigenSolver<Matrix>(m, "SelfAdjointEigenSolver");
  exposeTridiagonalization<Matrix>(m, "Tridiagonalization");

  // <Eigen/SparseCholesky>
  exposeSimplicialLDLT<SparseMatrix>(m, "SimplicialLDLT");
  exposeSimplicialLLT<SparseMatrix>(m, "SimplicialLLT");
  // <Eigen/SparseLU>
  exposeSparseLU<SparseMatrix>(m, "SparseLU");
  // <Eigen/SparseQR>
  exposeSparseQR<SparseMatrix>(m, "SparseQR");
#ifdef NANOEIGENPY_HAS_CHOLMOD
  // <Eigen/CholmodSupport>
  exposeCholmodSimplicialLLT<SparseMatrix>(m, "CholmodSimplicialLLT");
  exposeCholmodSimplicialLDLT<SparseMatrix>(m, "CholmodSimplicialLDLT");
  exposeCholmodSupernodalLLT<SparseMatrix>(m, "CholmodSupernodalLLT");
#endif
#ifdef NANOEIGENPY_HAS_ACCELERATE
  // <Eigen/AccelerateSupport>
  exposeAccelerate(m);
#endif

  // <Eigen/Geometry>
  exposeQuaternion<Scalar>(m, "Quaternion");
  exposeAngleAxis<Scalar>(m, "AngleAxis");

  // <Eigen/IterativeLinearSolvers>
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
