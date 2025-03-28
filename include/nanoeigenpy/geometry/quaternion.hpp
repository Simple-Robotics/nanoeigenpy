/// Copyright 2025 INRIA

#pragma once

#include "detail/rotation-base.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>

namespace nanoeigenpy {
namespace nb = nanobind;

/// Visitor for Eigen Quaternion types.
template <typename Quaternion>
struct QuaternionVisitor : nb::def_visitor<QuaternionVisitor<Quaternion>> {
  using Class = Quaternion;
  using QuaternionBase = Eigen::QuaternionBase<Quaternion>;
  static_assert(std::is_base_of_v<QuaternionBase, Quaternion>);
  using Scalar = typename QuaternionBase::Scalar;
  using Vector3 = typename QuaternionBase::Vector3;
  using Vector4 = typename QuaternionBase::Coefficients;
  using Matrix3 = typename QuaternionBase::Matrix3;
  using Coefficients = typename QuaternionBase::Coefficients;
  using AngleAxisType = typename QuaternionBase::AngleAxisType;

 public:
  template <typename... Ts>
  void execute(nb::class_<Class, Ts...>& cl) {
    using namespace nb::literals;
    // Inits
    cl.def(nb::init<>(), "Default constructor")
        .def(nb::init<Scalar, Scalar, Scalar, Scalar>(), "x"_a, "y"_a, "z"_a,
             "w"_a,
             "Initialize from coefficients.\n\n"
             "... note:: The order of coefficients is *w*, *x*, *y*, *z*. "
             "The [] operator numbers them differently, 0...4 for *x* *y* *z* "
             "*w*!")
        .def(nb::init<const AngleAxisType&>(), "aa"_a,
             "Initialize from an angle axis.\n"
             "\taa: angle axis object.")
        .def(nb::init<const Eigen::Ref<const Matrix3>>(), "R"_a,
             "Initialize from rotation matrix.\n"
             "\tR : a rotation matrix 3x3.")
        .def(nb::init<const Quaternion&>(), "other"_a,
             "Copy constructor.\n"
             "\tquat: a quaternion.")
        .def(
            "__init__",
            [](Quaternion* self, const Eigen::Ref<const Vector4>& v) {
              new (self) Quaternion(v[3], v[0], v[1], v[2]);
            },
            "v"_a,
            "Initialize from a 4D vector (xyzw).\n"
            "\tv : a 4D vector representing quaternion coefficients in the "
            "order xyzw.")
        .def(
            "__init__",
            [](Quaternion* self, const Eigen::Ref<const Eigen::Vector3d>& u,
               const Eigen::Ref<const Eigen::Vector3d>& v) {
              new (self) Quaternion();
              self->setFromTwoVectors(u, v);
            },
            "u"_a, "v"_a, "Initialize from two vectors u and v.")

        // Add property with x, y, z, w
        .def_prop_rw("x", &QuaternionVisitor::getCoeff<0>,
                     &QuaternionVisitor::setCoeff<0>, "The x coefficient.")
        .def_prop_rw("y", &QuaternionVisitor::getCoeff<1>,
                     &QuaternionVisitor::setCoeff<1>, "The y coefficient.")
        .def_prop_rw("z", &QuaternionVisitor::getCoeff<2>,
                     &QuaternionVisitor::setCoeff<2>, "The z coefficient.")
        .def_prop_rw("w", &QuaternionVisitor::getCoeff<3>,
                     &QuaternionVisitor::setCoeff<3>, "The w coefficient.")

        .def(
            "isApprox",
            [](const Quaternion& self, const Quaternion& other) -> bool {
              return self.isApprox(other,
                                   Eigen::NumTraits<Scalar>::dummy_precision());
            },
            "other"_a,
            "Returns true if *this is approximately equal to other using "
            "default precision.")
        .def(
            "isApprox",
            [](const Quaternion& self, const Quaternion& other,
               const Scalar& prec) -> bool {
              return self.isApprox(other, prec);
            },
            "other"_a, "prec"_a,
            "Returns true if *this is approximately equal to other, "
            "within the precision determined by prec.")

        // Methods
        .def(
            "coeffs",
            [](const Quaternion& self) -> const Vector4& {
              return self.coeffs();
            },
            "Returns a vector of the coefficients (x,y,z,w)",
            nb::rv_policy::reference_internal)
        .def(RotationBaseVisitor<Quaternion, 3>())
        .def("setFromTwoVectors", &setFromTwoVectors, "a"_a, "b"_a,
             nb::rv_policy::reference)
        .def("conjugate", &Quaternion::conjugate,
             "Returns the conjugated quaternion.\n"
             "The conjugate of a quaternion represents the opposite rotation.")
        .def("setIdentity", &Quaternion::setIdentity,
             "Set *this to the identity rotation.", nb::rv_policy::reference)
        .def("norm", &Quaternion::norm,
             "Returns the norm of the quaternion's coefficients.")
        .def("normalize", &Quaternion::normalize,
             "Normalizes the quaternion *this.", nb::rv_policy::reference)
        .def("normalized", &normalized, nb::rv_policy::take_ownership,
             "Returns a normalized copy of *this.")
        .def("squaredNorm", &Quaternion::squaredNorm,
             "Returns the squared norm of the quaternion's coefficients.")
        .def("dot", &Quaternion::template dot<Quaternion>, "other"_a,
             "Returns the dot product of *this with an other Quaternion.\n"
             "Geometrically speaking, the dot product of two unit quaternions "
             "corresponds to the cosine of half the angle between the two "
             "rotations.")
        .def("_transformVector", &Quaternion::_transformVector, "vector"_a,
             "Rotation of a vector by a quaternion.")
        .def(
            "vec", [](const Quaternion& self) -> Vector3 { return self.vec(); },
            "Returns a vector expression of the imaginary part (x,y,z).")
        .def("angularDistance",
             &Quaternion::template angularDistance<Quaternion>,
             "Returns the angle (in radian) between two rotations.")
        .def(
            "slerp",
            [](const Quaternion& self, const Scalar t, const Quaternion& other)
                -> Quaternion { return self.slerp(t, other); },
            "t"_a, "other"_a,
            "Returns the spherical linear interpolation between the two "
            "quaternions *this and other at the parameter t in [0;1].")

        // Operators
        .def(nb::self * nb::self)
        .def(nb::self *= nb::self, nb::rv_policy::none)
        .def(nb::self * Vector3())
        .def(nb::self == nb::self)
        .def("__eq__",
             [](const Quaternion& u, const Quaternion& v) -> bool {
               return u.coeffs() == v.coeffs();
             })
        .def("__ne__",
             [](const Quaternion& u, const Quaternion& v) -> bool {
               return u.coeffs() != v.coeffs();
             })
        .def("__abs__", &Quaternion::norm)
        .def("__len__", &QuaternionVisitor::__len__)
        .def("__setitem__", &QuaternionVisitor::__setitem__)
        .def("__getitem__", &QuaternionVisitor::__getitem__)
        .def("assign", &assign<Quaternion>,
             "Set *this from an quaternion quat and returns a reference to "
             "*this.",
             nb::rv_policy::reference)
        .def(
            "assign",
            [](Quaternion* self, const AngleAxisType& aa) -> Quaternion& {
              return (*self = aa);
            },
            nb::arg("aa"), nb::rv_policy::reference,
            "Set *this from an angle-axis and return a reference to self.")
        .def("__str__", &print)
        .def("__repr__", &print)

        .def_static("FromTwoVectors", &FromTwoVectors, "a"_a, "b"_a,
                    "Returns the quaternion which transforms a into b through "
                    "a rotation.",
                    nb::rv_policy::take_ownership)
        .def_static("Identity", &Identity,
                    "Returns a quaternion representing an identity rotation.",
                    nb::rv_policy::take_ownership);
  }

 private:
  static Quaternion* normalized(const Quaternion& self) {
    return new Quaternion(self.normalized());
  }

  static Quaternion& setFromTwoVectors(Quaternion& self,
                                       Eigen::Ref<const Vector3> a,
                                       Eigen::Ref<const Vector3> b) {
    return self.setFromTwoVectors(a, b);
  }

  template <int i>
  static Scalar getCoeff(Quaternion& self) {
    return self.coeffs()[i];
  }

  template <int i>
  static void setCoeff(Quaternion& self, Scalar value) {
    self.coeffs()[i] = value;
  }

  static int __len__() { return 4; }

  static Scalar __getitem__(const Quaternion& self, int idx) {
    if ((idx < 0) || (idx >= 4))
      throw nb::index_error("Index out of range [0, 3]");
    return self.coeffs()[idx];
  }

  static void __setitem__(Quaternion& self, int idx, const Scalar value) {
    if ((idx < 0) || (idx >= 4))
      throw nb::index_error("Index out of range [0, 3]");
    self.coeffs()[idx] = value;
  }

  template <typename OtherQuat>
  static Quaternion& assign(Quaternion& self, const OtherQuat& quat) {
    return self = quat;
  }

  static Quaternion* FromTwoVectors(const Eigen::Ref<const Vector3>& u,
                                    const Eigen::Ref<const Vector3>& v) {
    Quaternion* q = new Quaternion;
    q->setFromTwoVectors(u, v);
    return q;
  }

  static Quaternion* Identity() {
    Quaternion* q = new Quaternion;
    q->setIdentity();
    return q;
  }

  static std::string print(const Quaternion& self) {
    std::stringstream ss;
    ss << "(x,y,z,w) = " << self.coeffs().transpose() << std::endl;
    return ss.str();
  }

  template <typename Scalar>
  bool isApprox(
      const Quaternion& self, const Quaternion& other,
      const Scalar& prec = Eigen::NumTraits<Scalar>::dummy_precision()) {
    return self.isApprox(other, prec);
  }

 public:
  static void expose(nb::module_& m, const char* name) {
    nb::class_<Quaternion>(m, name).def(QuaternionVisitor());
  }
};

template <typename Scalar>
void exposeQuaternion(nb::module_& m, const char* name) {
  using Quaternion = Eigen::Quaternion<Scalar>;
  QuaternionVisitor<Quaternion>::expose(m, name);
}

}  // namespace nanoeigenpy
