/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include <Eigen/SVD>

namespace nanoeigenpy {
namespace nb = nanobind;
using namespace nb::literals;

struct SVDBaseVisitor : nb::def_visitor<SVDBaseVisitor> {
  template <typename Derived, typename... Ts>
  void execute(nb::class_<Derived, Ts...> &cl) {
    using MatrixType = typename Derived::MatrixType;
    using RealScalar = typename MatrixType::RealScalar;

    using SVDBase = Eigen::SVDBase<Derived>;
    static_assert(std::is_base_of_v<SVDBase, Derived>);

    cl.def(nb::init<>(), "Default constructor.")

        .def(
            "matrixU",
            [](Derived const &c) -> MatrixType { return c.matrixU(); },
            "Returns the U matrix")
        .def(
            "matrixV",
            [](Derived const &c) -> MatrixType { return c.matrixV(); },
            "Returns the U matrix")

        .def("singularValues", &SVDBase::singularValues,
             "For the SVD decomposition of a n-by-p matrix, letting "
             "\a m be the minimum of \a n and \a p, the returned vector "
             "has size \a m.  Singular values are always sorted in decreasing "
             "order.",
             nb::rv_policy::reference_internal)
        .def("nonzeroSingularValues", &SVDBase::nonzeroSingularValues,
             "Returns the number of singular values that are not exactly 0.")
        .def("rank", &SVDBase::rank,
             "Returns the rank of the matrix of which *this is the SVD.")

        .def(
            "setThreshold",
            [](Derived &c, RealScalar const &threshold) {
              return c.setThreshold(threshold);
            },
            "threshold"_a,
            "Allows to prescribe a threshold to be used by certain methods, "
            "such as rank(), who need to determine when pivots are to be "
            "considered nonzero. This is not used for the SVD decomposition "
            "itself.\n\n"
            "When it needs to get the threshold value, Eigen calls "
            "threshold().",
            nb::rv_policy::reference)
        .def(
            "setThreshold",
            [](Derived &c) { return c.setThreshold(Eigen::Default); },
            "Allows to come back to the default behavior, letting Eigen use "
            "its default formula for determining the threshold.",
            nb::rv_policy::reference)
        .def("threshold", &SVDBase::threshold,
             "Returns the threshold that will be used by certain methods such "
             "as rank().")

        .def("computeU", &SVDBase::computeU,
             "Returns true if U (full or thin) is asked for in this SVD "
             "decomposition")
        .def("computeV", &SVDBase::computeV,
             "Returns true if V (full or thin) is asked for in this SVD "
             "decomposition")

        .def("rows", &SVDBase::rows,
             "Returns the number of rows of the matrix.")
        .def("cols", &SVDBase::cols,
             "Returns the number of cols of the matrix.")

        .def("info", &SVDBase::info,
             "Reports whether previous computation was successful.");
  }
};

}  // namespace nanoeigenpy
