/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include <nanobind/operators.h>

namespace nanoeigenpy {
namespace nb = nanobind;

template <typename Scalar>
bool isApprox(
    const Eigen::Translation<Scalar, Eigen::Dynamic>& r,
    const Eigen::Translation<Scalar, Eigen::Dynamic>& other,
    const Scalar& prec = Eigen::NumTraits<Scalar>::dummy_precision()) {
  return r.isApprox(other, prec);
}

template <typename Scalar>
void exposeTranslation(nb::module_ m, const char* name) {
  using namespace nb::literals;
  using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using LinearMatrixType =
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using Translation = Eigen::Translation<Scalar, Eigen::Dynamic>;

  if (check_registration_alias<Translation>(m)) {
    return;
  }
  nb::class_<Translation>(m, name, "Represents a translation transformation.")
      .def(nb::init<>(), "Default constructor")
      .def(nb::init<const Scalar&, const Scalar&>(), "sx"_a, "sy"_a)
      .def(nb::init<const Scalar&, const Scalar&, const Scalar&>(), "sx"_a,
           "sy"_a, "sz"_a)
      .def(nb::init<const VectorType&>(), "vector"_a)
      .def(nb::init<const Translation&>(), "copy"_a, "Copy constructor.")

      .def_prop_rw(
          "x",
          [](const Translation& self) -> Scalar {
            if (self.vector().size() < 1) {
              throw std::out_of_range(
                  "Translation must have at least 1 dimension for x");
            }
            return self.vector()[0];
          },
          [](Translation& self, const Scalar& value) {
            if (self.vector().size() < 1) {
              throw std::out_of_range(
                  "Translation must have at least 1 dimension for x");
            }
            self.vector()[0] = value;
          },
          "The x-translation")
      .def_prop_rw(
          "y",
          [](const Translation& self) -> Scalar {
            if (self.vector().size() < 2) {
              throw std::out_of_range(
                  "Translation must have at least 2 dimensions for y");
            }
            return self.vector()[1];
          },
          [](Translation& self, const Scalar& value) {
            if (self.vector().size() < 2) {
              throw std::out_of_range(
                  "Translation must have at least 2 dimensions for y");
            }
            self.vector()[1] = value;
          },
          "The y-translation")
      .def_prop_rw(
          "z",
          [](const Translation& self) -> Scalar {
            if (self.vector().size() < 3) {
              throw std::out_of_range(
                  "Translation must have at least 3 dimensions for z");
            }
            return self.vector()[2];
          },
          [](Translation& self, const Scalar& value) {
            if (self.vector().size() < 3) {
              throw std::out_of_range(
                  "Translation must have at least 3 dimensions for z");
            }
            self.vector()[2] = value;
          },
          "The z-translation")

      .def(
          "vector",
          [](const Translation& self) -> const VectorType& {
            return self.vector();
          },
          nb::rv_policy::reference_internal, "Returns the translation vector")

      .def(
          "translation",
          [](const Translation& self) -> const VectorType& {
            return self.translation();
          },
          nb::rv_policy::reference_internal,
          "Returns the translation vector (alias for vector)")

      .def("inverse", &Translation::inverse,
           "Returns the inverse translation (opposite)")

      .def(
          "isApprox",
          [](const Translation& r, const Translation& other,
             const Scalar& prec) -> bool { return isApprox(r, other, prec); },
          "other"_a, "prec"_a,
          "Returns true if *this is approximately equal to other, "
          "within the precision determined by prec.")
      .def(
          "isApprox",
          [](const Translation& r, const Translation& other) -> bool {
            return isApprox(r, other);
          },
          "other"_a,
          "Returns true if *this is approximately equal to other, "
          "within the default precision.")

      .def(
          "__mul__",
          [](const Translation& self, const Translation& other) -> Translation {
            return self * other;
          },
          "other"_a, "Concatenates two translations")

      .def(IdVisitor());
}

}  // namespace nanoeigenpy
