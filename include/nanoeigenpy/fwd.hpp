/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/config.hpp"
#include "nanoeigenpy/id.hpp"
#include "nanoeigenpy/utils/helpers.hpp"
#include <nanobind/nanobind.h>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Geometry>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Jacobi>

namespace nanoeigenpy {
namespace nb = nanobind;
}  // namespace nanoeigenpy
