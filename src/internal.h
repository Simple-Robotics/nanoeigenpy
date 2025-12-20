#include <nanobind/nanobind.h>
#include <Eigen/Core>

namespace nb = nanobind;

using Scalar = double;
static constexpr int Options = Eigen::ColMajor;
using Matrix = Eigen::Matrix<Scalar, -1, -1, Options>;
using Vector = Eigen::Matrix<Scalar, -1, 1>;
