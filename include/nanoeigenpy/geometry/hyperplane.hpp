/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include <nanobind/operators.h>

namespace nanoeigenpy {
namespace nb = nanobind;

template <typename Scalar>
bool isApprox(
    const Eigen::Hyperplane<Scalar, Eigen::Dynamic> &h,
    const Eigen::Hyperplane<Scalar, Eigen::Dynamic> &other,
    const Scalar &prec = Eigen::NumTraits<Scalar>::dummy_precision()) {
  return h.isApprox(other, prec);
}

template <typename Scalar>
void exposeHyperplane(nb::module_ m, const char *name) {
  using namespace nb::literals;
  using Hyperplane = Eigen::Hyperplane<Scalar, Eigen::Dynamic>;
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using Parameterized = Eigen::ParametrizedLine<Scalar, Eigen::Dynamic>;

  if (check_registration_alias<Hyperplane>(m)) {
    return;
  }
  nb::class_<Hyperplane>(m, name,
                         "A hyperplane is an affine subspace of "
                         "dimension n-1 in a space of dimension n.")
      .def(nb::init<>(), "Default constructor")
      .def(nb::init<const Hyperplane &>(), "copy"_a, "Copy constructor.")
      .def(nb::init<Eigen::DenseIndex>(), "dim"_a,
           "Constructs a dynamic-size hyperplane with dim the dimension"
           "of the ambient space.")
      .def(nb::init<const VectorType &, const VectorType &>(), "n"_a, "e"_a,
           "Construct a plane from its normal \a n and a point \a e onto the "
           "plane.")
      .def(nb::init<const VectorType &, const Scalar &>(), "n"_a, "d"_a,
           "Constructs a plane from its normal n and distance to the origin d"
           "such that the algebraic equation of the plane is \f$ n dot x + d = "
           "0 \f$.")
      .def(nb::init<const Parameterized &>(), "parametrized"_a,
           "Constructs a hyperplane passing through the parametrized line \a "
           "parametrized."
           "If the dimension of the ambient space is greater than 2, then "
           "there isn't uniqueness,"
           "so an arbitrary choice is made.")
      .def_static(
          "Through",
          [](const VectorType &p0, const VectorType &p1) -> Hyperplane {
            return Hyperplane::Through(p0, p1);
          },
          "p0"_a, "p1"_a,
          "Constructs a hyperplane passing through the two points. "
          "If the dimension of the ambient space is greater than 2, then "
          "there isn't uniqueness, so an arbitrary choice is made.")
      .def_static(
          "Through",
          [](const VectorType &p0, const VectorType &p1,
             const VectorType &p2) -> Hyperplane {
            if (p0.size() != 3 || p1.size() != 3 || p2.size() != 3) {
              throw std::invalid_argument(
                  "Through with 3 points requires 3D vectors");
            }

            Hyperplane result(p0.size());
            VectorType v0 = p2 - p0;
            VectorType v1 = p1 - p0;

            VectorType normal(3);
            normal[0] = v0[1] * v1[2] - v0[2] * v1[1];
            normal[1] = v0[2] * v1[0] - v0[0] * v1[2];
            normal[2] = v0[0] * v1[1] - v0[1] * v1[0];

            RealScalar norm = normal.norm();
            if (norm <= v0.norm() * v1.norm() *
                            Eigen::NumTraits<RealScalar>::epsilon()) {
              Eigen::Matrix<Scalar, 2, 3> m;
              m.row(0) = v0.transpose();
              m.row(1) = v1.transpose();
              Eigen::JacobiSVD<Eigen::Matrix<Scalar, 2, 3>> svd(
                  m, Eigen::ComputeFullV);
              result.normal() = svd.matrixV().col(2);
            } else {
              result.normal() = normal / norm;
            }

            result.offset() = -p0.dot(result.normal());
            return result;
          },
          "p0"_a, "p1"_a, "p2"_a,
          "Constructs a hyperplane passing through the three points. "
          "The dimension of the ambient space is required to be exactly 3.")

      .def("dim", &Hyperplane::dim,
           "Returns the dimension in which the plane holds.")
      .def("normalize", &Hyperplane::normalize, "Normalizes *this.")

      .def("signedDistance", &Hyperplane::signedDistance, "p"_a,
           "Returns the signed distance between the plane *this and a point p.")
      .def("absDistance", &Hyperplane::absDistance, "p"_a,
           "Returns the absolute distance between the plane *this and a point "
           "p.")
      .def("projection", &Hyperplane::projection, "p"_a,
           "Returns the projection of a point \a p onto the plane *this.")

      .def(
          "normal",
          [](const Hyperplane &self) -> VectorType {
            return VectorType(self.normal());
          },
          "Returns a constant reference to the unit normal vector of the "
          "plane, "
          "which corresponds to the linear part of the implicit equation.")
      .def(
          "offset",
          [](const Hyperplane &self) -> const Scalar & {
            return self.offset();
          },
          "Returns the distance to the origin, which is also the constant "
          "term of the implicit equation.",
          nb::rv_policy::reference_internal)
      .def(
          "coeffs", [](const Hyperplane &self) { return self.coeffs(); },
          "Returns a constant reference to the coefficients c_i of the plane "
          "equation: "
          "\f$ c_0*x_0 + ... + c_{d-1}*x_{d-1} + c_d = 0 \f$.",
          nb::rv_policy::reference_internal)

      .def(
          "intersection",
          [](const Hyperplane &self, const Hyperplane &other) -> VectorType {
            if (self.dim() != 2 || other.dim() != 2) {
              throw std::invalid_argument(
                  "intersection requires 2D hyperplanes");
            }

            Scalar det = self.coeffs().coeff(0) * other.coeffs().coeff(1) -
                         self.coeffs().coeff(1) * other.coeffs().coeff(0);

            if (Eigen::internal::isMuchSmallerThan(det, Scalar(1))) {
              if (Eigen::numext::abs(self.coeffs().coeff(1)) >
                  Eigen::numext::abs(self.coeffs().coeff(0))) {
                VectorType result(2);
                result[0] = self.coeffs().coeff(1);
                result[1] = -self.coeffs().coeff(2) / self.coeffs().coeff(1) -
                            self.coeffs().coeff(0);
                return result;
              } else {
                VectorType result(2);
                result[0] = -self.coeffs().coeff(2) / self.coeffs().coeff(0) -
                            self.coeffs().coeff(1);
                result[1] = self.coeffs().coeff(0);
                return result;
              }
            } else {
              Scalar invdet = Scalar(1) / det;
              VectorType result(2);
              result[0] =
                  invdet * (self.coeffs().coeff(1) * other.coeffs().coeff(2) -
                            other.coeffs().coeff(1) * self.coeffs().coeff(2));
              result[1] =
                  invdet * (other.coeffs().coeff(0) * self.coeffs().coeff(2) -
                            self.coeffs().coeff(0) * other.coeffs().coeff(2));
              return result;
            }
          },
          "other"_a, "Returns the intersection of *this with \a other.")

      .def(
          "isApprox",
          [](const Hyperplane &aa, const Hyperplane &other,
             const Scalar &prec) -> bool { return isApprox(aa, other, prec); },
          "other"_a, "prec"_a,
          "Returns true if *this is approximately equal to other, "
          "within the precision determined by prec.")
      .def(
          "isApprox",
          [](const Hyperplane &aa, const Hyperplane &other) -> bool {
            return isApprox(aa, other);
          },
          "other"_a,
          "Returns true if *this is approximately equal to other, "
          "within the default precision.")

      .def(IdVisitor());
}

}  // namespace nanoeigenpy
