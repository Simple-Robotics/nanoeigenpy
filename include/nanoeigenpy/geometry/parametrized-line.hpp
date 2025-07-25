/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include <nanobind/operators.h>

namespace nanoeigenpy {
namespace nb = nanobind;

template <typename Scalar>
bool isApprox(
    const Eigen::ParametrizedLine<Scalar, Eigen::Dynamic> &h,
    const Eigen::ParametrizedLine<Scalar, Eigen::Dynamic> &other,
    const Scalar &prec = Eigen::NumTraits<Scalar>::dummy_precision()) {
  return h.isApprox(other, prec);
}

template <typename Scalar>
void exposeParametrizedLine(nb::module_ m, const char *name) {
  using namespace nb::literals;
  using ParametrizedLine = Eigen::ParametrizedLine<Scalar, Eigen::Dynamic>;
  using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using Hyperplane = Eigen::Hyperplane<Scalar, Eigen::Dynamic>;

  if (check_registration_alias<ParametrizedLine>(m)) {
    return;
  }
  nb::class_<ParametrizedLine>(m, name, "Parametrized line.")
      .def(nb::init<>(), "Default constructor")
      .def(nb::init<const ParametrizedLine &>(), "copy"_a, "Copy constructor.")
      .def(nb::init<Eigen::DenseIndex>(), "dim"_a,
           "Constructs a dynamic-size line with \a _dim the dimension"
           "of the ambient space.")
      .def(nb::init<const VectorType &, const VectorType &>(), "origin"_a,
           "direction"_a,
           "Initializes a parametrized line of direction \a direction and "
           "origin \a origin.")
      .def(nb::init<const ParametrizedLine &>(), "copy"_a, "Copy constructor.")
      .def(
          "__init__",
          [](ParametrizedLine *self, const Hyperplane &hyperplane) {
            if (hyperplane.dim() != 2) {
              throw std::invalid_argument(
                  "ParametrizedLine from Hyperplane requires ambient space "
                  "dimension 2");
            }

            VectorType normal = hyperplane.normal().eval();
            VectorType direction(2);
            direction[0] = -normal[1];
            direction[1] = normal[0];

            VectorType origin = -normal * hyperplane.offset();

            new (self) ParametrizedLine(origin, direction);
          },
          "hyperplane"_a,
          "Constructs a parametrized line from a 2D hyperplane.")
      .def_static(
          "Through",
          [](const VectorType &p0, const VectorType &p1) -> ParametrizedLine {
            return ParametrizedLine::Through(p0, p1);
          },
          "p0"_a, "p1"_a,
          "Constructs a parametrized line going from \a p0 to \a p1.")

      .def("dim", &ParametrizedLine::dim,
           "Returns the dimension in which the line holds.")

      .def(
          "origin",
          [](const ParametrizedLine &self) -> const VectorType & {
            return self.origin();
          },
          nb::rv_policy::reference_internal,
          "Returns the origin of the parametrized line.")
      .def(
          "direction",
          [](const ParametrizedLine &self) -> const VectorType & {
            return self.direction();
          },
          nb::rv_policy::reference_internal,
          "Returns the direction of the parametrized line.")

      .def("squaredDistance", &ParametrizedLine::squaredDistance, "p"_a,
           "Returns the squared distance of a point \a p to "
           "its projection onto the line *this.")
      .def("distance", &ParametrizedLine::distance, "p"_a,
           "Returns the distance of a point \a p to "
           "its projection onto the line *this.")

      .def("projection", &ParametrizedLine::projection, "p"_a,
           "Returns the projection of a point \a p onto the line *this.")
      .def("pointAt", &ParametrizedLine::pointAt, "t"_a)

      .def(
          "intersectionParameter",
          [](const ParametrizedLine &self, const Hyperplane &hyperplane)
              -> Scalar { return self.intersectionParameter(hyperplane); },
          "hyperplane"_a,
          "Returns the parameter value of the intersection between this line "
          "and the given hyperplane.")

      .def(
          "intersection",
          [](const ParametrizedLine &self, const Hyperplane &hyperplane)
              -> Scalar { return self.intersection(hyperplane); },
          "hyperplane"_a,
          "Deprecated: use intersectionParameter(). Returns the parameter "
          "value of the intersection.")

      .def(
          "intersectionPoint",
          [](const ParametrizedLine &self, const Hyperplane &hyperplane)
              -> VectorType { return self.intersectionPoint(hyperplane); },
          "hyperplane"_a,
          "Returns the point of the intersection between this line and the "
          "given hyperplane.")

      .def(
          "isApprox",
          [](const ParametrizedLine &aa, const ParametrizedLine &other,
             const Scalar &prec) -> bool { return isApprox(aa, other, prec); },
          "other"_a, "prec"_a,
          "Returns true if *this is approximately equal to other, "
          "within the precision determined by prec.")
      .def(
          "isApprox",
          [](const ParametrizedLine &aa, const ParametrizedLine &other)
              -> bool { return isApprox(aa, other); },
          "other"_a,
          "Returns true if *this is approximately equal to other, "
          "within the default precision.")

      .def(IdVisitor());
}

}  // namespace nanoeigenpy
