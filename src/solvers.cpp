#include "nanoeigenpy/solvers.hpp"

#include "./internal.h"

using namespace nanoeigenpy;

void exposeSolvers(nb::module_& m) {
  exposeIdentityPreconditioner<Scalar>(m, "IdentityPreconditioner");
  exposeDiagonalPreconditioner<Scalar>(m, "DiagonalPreconditioner");
#if EIGEN_VERSION_AT_LEAST(3, 3, 5)
  exposeLeastSquareDiagonalPreconditioner<Scalar>(
      m, "LeastSquareDiagonalPreconditioner");
#endif
  exposeMINRESSolver<Matrix>(m, "MINRES");

  // Solvers
  using Eigen::ConjugateGradient;
  using Eigen::IdentityPreconditioner;
  using Eigen::LeastSquareDiagonalPreconditioner;
  using Eigen::LeastSquaresConjugateGradient;
  using Eigen::Lower;
  using Eigen::Upper;

  exposeConjugateGradient<ConjugateGradient<Matrix, Lower | Upper>>(
      m, "ConjugateGradient");

  exposeLeastSquaresConjugateGradient<LeastSquaresConjugateGradient<
      Matrix, LeastSquareDiagonalPreconditioner<Scalar>>>(
      m, "LeastSquaresConjugateGradient");

  using IdentityConjugateGradient =
      ConjugateGradient<Matrix, Lower | Upper, IdentityPreconditioner>;
  exposeConjugateGradient<IdentityConjugateGradient>(
      m, "IdentityConjugateGradient");
}
