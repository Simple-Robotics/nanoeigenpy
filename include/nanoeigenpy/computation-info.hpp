/// Copyright 2025 INRIA

#pragma once

#include <nanobind/nanobind.h>
#include <Eigen/Core>

namespace nanoeigenpy {
namespace nb = nanobind;
inline void exposeComputationInfo(nb::module_ m) {
  nb::enum_<Eigen::ComputationInfo>(m, "ComputationInfo")
      .value("Success", Eigen::Success, "Computation was successful.")
      .value("NumericalIssue", Eigen::NumericalIssue,
             "The provided data did not satisfy the prerequisites.")
      .value("NoConvergence", Eigen::NoConvergence,
             "Iterative procedure did not converge.")
      .value("InvalidInput", Eigen::InvalidInput,
             "The inputs are invalid, or the algorithm has been improperly "
             "called. "
             "When assertions are enabled, such errors trigger an assert.")
      .export_values();
}
}  // namespace nanoeigenpy
