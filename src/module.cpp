/// Copyright 2025 INRIA

#include <nanobind/stl/string.h>

#include "nanoeigenpy/fwd.hpp"
#include "nanoeigenpy/decompositions.hpp"
#include "nanoeigenpy/geometry.hpp"
#include "nanoeigenpy/solvers.hpp"
#include "nanoeigenpy/constants.hpp"
#include "nanoeigenpy/utils/is-approx.hpp"

namespace nb = nanobind;

using Scalar = double;
static constexpr int Options = Eigen::ColMajor;
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Options>;
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

using namespace nanoeigenpy;

// Matrix types
using SparseMatrix = Eigen::SparseMatrix<Scalar, Options>;

using Eigen::ColPivHouseholderQRPreconditioner;
using Eigen::FullPivHouseholderQRPreconditioner;
using Eigen::HouseholderQRPreconditioner;
using Eigen::JacobiSVD;
using Eigen::NoQRPreconditioner;

using ColPivHhJacobiSVD = JacobiSVD<Matrix, ColPivHouseholderQRPreconditioner>;
using FullPivHhJacobiSVD =
    JacobiSVD<Matrix, FullPivHouseholderQRPreconditioner>;
using HhJacobiSVD = JacobiSVD<Matrix, HouseholderQRPreconditioner>;
using NoPrecondJacobiSVD = JacobiSVD<Matrix, NoQRPreconditioner>;

using SparseQR = Eigen::SparseQR<SparseMatrix, Eigen::COLAMDOrdering<int>>;
using SparseLU = Eigen::SparseLU<SparseMatrix>;
using SCMatrix = typename SparseLU::SCMatrix;
using StorageIndex = typename Matrix::StorageIndex;
#if EIGEN_VERSION_AT_LEAST(5, 0, 0)
using MappedSparseMatrix =
    Eigen::Map<Eigen::SparseMatrix<Scalar, Options, StorageIndex>>;
#else
using MappedSparseMatrix =
    Eigen::MappedSparseMatrix<Scalar, Options, StorageIndex>;
#endif

NB_MAKE_OPAQUE(ColPivHhJacobiSVD)
NB_MAKE_OPAQUE(FullPivHhJacobiSVD)
NB_MAKE_OPAQUE(HhJacobiSVD)
NB_MAKE_OPAQUE(NoPrecondJacobiSVD)

NB_MAKE_OPAQUE(Eigen::SparseQRMatrixQReturnType<SparseQR>)
NB_MAKE_OPAQUE(Eigen::SparseQRMatrixQTransposeReturnType<SparseQR>)
NB_MAKE_OPAQUE(Eigen::SparseLUMatrixLReturnType<SCMatrix>)
NB_MAKE_OPAQUE(Eigen::SparseLUMatrixUReturnType<SCMatrix, MappedSparseMatrix>)

NB_MAKE_OPAQUE(Eigen::LLT<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::LDLT<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::FullPivLU<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::PartialPivLU<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::ColPivHouseholderQR<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::FullPivHouseholderQR<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::HouseholderQR<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::BDCSVD<Eigen::MatrixXd>)
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

// Utils
std::string printEigenVersion(const char* delim = ".") {
  std::ostringstream oss;
  oss << EIGEN_WORLD_VERSION << delim << EIGEN_MAJOR_VERSION << delim
      << EIGEN_MINOR_VERSION;
  return oss.str();
}

// Module
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
  exposeJacobiSVD<ColPivHhJacobiSVD>(m, "ColPivHhJacobiSVD");
  exposeJacobiSVD<FullPivHhJacobiSVD>(m, "FullPivHhJacobiSVD");
  exposeJacobiSVD<HhJacobiSVD>(m, "HhJacobiSVD");
  exposeJacobiSVD<NoPrecondJacobiSVD>(m, "NoPrecondJacobiSVD");
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
  exposeHyperplane<Scalar>(m, "Hyperplane");
  exposeParametrizedLine<Scalar>(m, "ParametrizedLine");
  exposeRotation2D<Scalar>(m, "Rotation2D");
  exposeUniformScaling<Scalar>(m, "UniformScaling");
  exposeTranslation<Scalar>(m, "Translation");

  // <Eigen/Jacobi>
  exposeJacobiRotation<Scalar>(m, "JacobiRotation");

  // <Eigen/IterativeLinearSolvers>
  nb::module_ solvers =
      m.def_submodule("solvers", "Iterative linear solvers in Eigen.");
  exposeIdentityPreconditioner<Scalar>(solvers, "IdentityPreconditioner");
  exposeDiagonalPreconditioner<Scalar>(solvers, "DiagonalPreconditioner");
#if EIGEN_VERSION_AT_LEAST(3, 3, 5)
  exposeLeastSquareDiagonalPreconditioner<Scalar>(
      solvers, "LeastSquareDiagonalPreconditioner");
#endif

  using Eigen::Lower;

  using Eigen::BiCGSTAB;
  using Eigen::ConjugateGradient;
  using Eigen::DiagonalPreconditioner;
  using Eigen::IdentityPreconditioner;
  using Eigen::LeastSquareDiagonalPreconditioner;
  using Eigen::LeastSquaresConjugateGradient;
  using Eigen::MINRES;

  using IdentityConjugateGradient =
      ConjugateGradient<Matrix, Lower, IdentityPreconditioner>;
  using IdentityLeastSquaresConjugateGradient =
      LeastSquaresConjugateGradient<Matrix, IdentityPreconditioner>;
  using DiagonalLeastSquaresConjugateGradient =
      LeastSquaresConjugateGradient<Matrix, DiagonalPreconditioner<Scalar>>;
  using IdentityBiCGSTAB = BiCGSTAB<Matrix, IdentityPreconditioner>;
  using DiagonalMINRES = MINRES<Matrix, Lower, DiagonalPreconditioner<Scalar>>;

  exposeConjugateGradient<ConjugateGradient<Matrix, Lower>>(
      solvers, "ConjugateGradient");
  exposeConjugateGradient<IdentityConjugateGradient>(
      solvers, "IdentityConjugateGradient");
  exposeLeastSquaresConjugateGradient<LeastSquaresConjugateGradient<Matrix>>(
      solvers, "LeastSquaresConjugateGradient");
  exposeLeastSquaresConjugateGradient<IdentityLeastSquaresConjugateGradient>(
      solvers, "IdentityLeastSquaresConjugateGradient");
  exposeLeastSquaresConjugateGradient<DiagonalLeastSquaresConjugateGradient>(
      solvers, "DiagonalLeastSquaresConjugateGradient");
  exposeMINRES<MINRES<Matrix, Lower>>(solvers, "MINRES");
  exposeMINRES<DiagonalMINRES>(solvers, "DiagonalMINRES");
  exposeBiCGSTAB<BiCGSTAB<Matrix>>(solvers, "BiCGSTAB");
  exposeBiCGSTAB<IdentityBiCGSTAB>(solvers, "IdentityBiCGSTAB");

  exposeIncompleteLUT<SparseMatrix>(solvers, "IncompleteLUT");
  exposeIncompleteCholesky<SparseMatrix>(solvers, "IncompleteCholesky");

  // Utils
  exposeIsApprox<double>(m);
  exposeIsApprox<std::complex<double>>(m);

  m.attr("__version__") = NANOEIGENPY_VERSION;
  m.attr("__eigen_version__") = printEigenVersion();
  m.attr("__eigen_max_align_bytes__") = EIGEN_MAX_ALIGN_BYTES;

  m.def("SimdInstructionSetsInUse", &Eigen::SimdInstructionSetsInUse,
        "Get the set of SIMD instructions used in Eigen when this module was "
        "compiled.");
}
