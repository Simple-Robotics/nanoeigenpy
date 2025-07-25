/// Copyright 2025 INRIA

#pragma once

#include <nanobind/nanobind.h>
#include <Eigen/Core>

namespace nanoeigenpy {
namespace nb = nanobind;
inline void exposeConstants(nb::module_ m) {
  nb::enum_<Eigen::ComputationInfo>(m, "ComputationInfo")
      .value("Success", Eigen::Success, "Computation was successful.")
      .value("NumericalIssue", Eigen::NumericalIssue,
             "The provided data did not satisfy the prerequisites.")
      .value("NoConvergence", Eigen::NoConvergence,
             "Iterative procedure did not converge.")
      .value("InvalidInput", Eigen::InvalidInput,
             "The inputs are invalid, or the algorithm has been improperly "
             "called. "
             "When assertions are enabled, such errors trigger an assert.");
  using Eigen::DecompositionOptions;
#define _c(name) value(#name, DecompositionOptions::name)
  nb::enum_<DecompositionOptions>(m, "DecompositionOptions")
      ._c(ComputeFullU)
      ._c(ComputeThinU)
      ._c(ComputeFullV)
      ._c(ComputeThinV)
      ._c(EigenvaluesOnly)
      ._c(ComputeEigenvectors)
      ._c(Ax_lBx)
      ._c(ABx_lx)
      ._c(BAx_lx);
#undef _c
  using Eigen::TransformTraits;
#define _c(name) value(#name, TransformTraits::name)
  nb::enum_<TransformTraits>(m, "TransformTraits")
      ._c(Isometry)
      ._c(Affine)
      ._c(AffineCompact)
      ._c(Projective);
#undef _c
}
}  // namespace nanoeigenpy
