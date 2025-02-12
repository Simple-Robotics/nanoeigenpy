/// Copyright 2025 INRIA
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "nanoeigenpy/decompositions/ldlt.hpp"
#include "nanoeigenpy/decompositions/llt.hpp"
#include "nanoeigenpy/geometry/quaternion.hpp"

using namespace nanoeigenpy;

using Scalar = double;
static constexpr int Options = Eigen::ColMajor;
using Matrix = Eigen::Matrix<Scalar, -1, -1, Options>;
using Vector = Eigen::Matrix<Scalar, -1, 1>;
using Quaternion = Eigen::Quaternion<Scalar, Options>;

NB_MAKE_OPAQUE(Eigen::LLT<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::LDLT<Eigen::MatrixXd>)

NB_MODULE(nanoeigenpy, m) {
  exposeLLTSolver<Matrix>(m, "LLT");
  exposeLDLTSolver<Matrix>(m, "LDLT");
  QuaternionVisitor<Quaternion>::expose(m, "Quaternion");
}
