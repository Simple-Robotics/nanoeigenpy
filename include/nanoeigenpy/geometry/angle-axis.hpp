#pragma once

#include "detail/rotation-base.hpp"
#include <nanobind/operators.h>

namespace nanoeigenpy {
namespace nb = nanobind;

template <typename Scalar>
bool isApprox(const Eigen::AngleAxis<Scalar> &aa, 
           const Eigen::AngleAxis<Scalar> &other,
           const Scalar &prec = Eigen::NumTraits<Scalar>::dummy_precision()) 
{
     return aa.isApprox(other, prec);
}

template <typename Scalar>
void exposeAngleAxis(nb::module_ m, const char *name) {
  using namespace nb::literals;
  using AngleAxis = Eigen::AngleAxis<Scalar>;
  using Quaternion = typename Eigen::Quaternion<Scalar, 0>;
  using Vector3 = typename AngleAxis::Vector3;
  using Matrix3 = typename AngleAxis::Matrix3;

  auto cl = nb::class_<AngleAxis>(m, name, 
            "AngleAxis representation of a rotation.\n\n")

            .def(nb::init<>(),
            "Default constructor")
            .def(nb::init<const Scalar &, Vector3>(), 
            nb::arg("angle"), nb::arg("axis"), 
                "Initialize from angle and axis.")
            .def(nb::init<Matrix3>(), 
            nb::arg("R"),
            "Initialize from a rotation matrix.")
            .def(nb::init<Quaternion>(), 
            nb::arg("quaternion"),
            "Initialize from a quaternion.")
            .def(nb::init<AngleAxis>(), 
            nb::arg("copy"),
            "Copy constructor.")

            
            .def_prop_rw(
                "angle", [](const AngleAxis &aa) { return aa.angle(); },
                [](AngleAxis &aa, Scalar a) { aa.angle() = a; },
                "The rotation angle.", 
                nb::rv_policy::reference_internal)
            .def_prop_rw(
                "axis", [](const AngleAxis &aa) { return aa.axis(); },
                [](AngleAxis &aa, const Vector3 &v) { aa.axis() = v; },
                "The rotation axis.")

            .def(RotationBaseVisitor<AngleAxis, 3>())
            .def("fromRotationMatrix",
                &AngleAxis::template fromRotationMatrix<Matrix3>,
                nb::arg("self"), nb::arg("rotation matrix"),
                "Sets *this from a 3x3 rotation matrix", 
                nb::rv_policy::reference_internal)

            .def("isApprox",                                                                        
                [](const AngleAxis &aa, const AngleAxis &other, const Scalar &prec) 
                -> bool { return isApprox(aa, other, prec); },
                nb::arg("other"), nb::arg("rec"),
                "Returns true if *this is approximately equal to other, "
                "within the precision determined by prec.")

            .def(nb::self * nb::self)

            ;
}

}  // namespace nanoeigenpy


// TODO

// Differences between angle-axis and decompositions files:
// Remove the isApprox and make it clean
// bp::implicitly_convertible -> Let it for later (eventually Justin work)
// Manage the overload of isApprox (cf the macro in eigenpy) (try with quaternion implemented too, add if needed (probably))
// angle and axis functions: check if correct from eigenpy (probably not)
// See if clean to factorize with RotationBaseVisitor
// How to define the operators

