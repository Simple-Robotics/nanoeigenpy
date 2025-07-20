/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include <Eigen/IterativeLinearSolvers>

namespace nanoeigenpy {
namespace nb = nanobind;

template <typename Preconditioner>
struct PreconditionerBaseVisitor
    : nb::def_visitor<PreconditionerBaseVisitor<Preconditioner>> {
  using MatrixType = Eigen::MatrixXd;
  using VectorType = Eigen::VectorXd;

  template <typename... Ts>
  void execute(nb::class_<Preconditioner, Ts...>& cl) {
    using namespace nb::literals;
    cl.def(nb::init<>())
        .def(nb::init<MatrixType>(), "A"_a)
        .def("info", &Preconditioner::info,
             "Returns success if the Preconditioner has been well initialized.")
        .def("solve", &solve, "b"_a,
             "Returns the solution A * z = b where the preconditioner is an "
             "estimate of A^-1.")

        .def("compute", &Preconditioner::template compute<MatrixType>, "mat"_a,
             "Initialize the preconditioner from the matrix value.",
             nb::rv_policy::reference)
        .def("factorize", &Preconditioner::template factorize<MatrixType>,
             "mat"_a,
             "Initialize the preconditioner from the matrix value, i.e "
             "factorize the mat given as input to approximate its inverse.",
             nb::rv_policy::reference);
  }

 private:
  static VectorType solve(const Preconditioner& self, const VectorType& vec) {
    return self.solve(vec);
  }
};

template <typename Scalar>
struct DiagonalPreconditionerVisitor
    : PreconditionerBaseVisitor<DiagonalPreconditionerVisitor<Scalar>> {
  using Preconditioner = Eigen::DiagonalPreconditioner<Scalar>;

  template <typename... Ts>
  void execute(nb::class_<Scalar, Ts...>& cl) {
    using namespace nb::literals;
    cl.def(PreconditionerBaseVisitor<Preconditioner>())
        .def("rows", &Preconditioner::rows,
             "Returns the number of rows in the preconditioner.")
        .def("cols", &Preconditioner::rows,
             "Returns the number of cols in the preconditioner.");
  }

  static void expose(nb::module_& m, const char* name) {
    if (check_registration_alias<Preconditioner>(m)) {
      return;
    }
    nb::class_<Preconditioner>(m, name).def(IdVisitor());
  }
};

template <typename Scalar>
void exposeDiagonalPreconditioner(nb::module_& m, const char* name) {
  DiagonalPreconditionerVisitor<Scalar>::expose(m, name);
}

#if EIGEN_VERSION_AT_LEAST(3, 3, 5)
template <typename Scalar>
struct LeastSquareDiagonalPreconditionerVisitor
    : PreconditionerBaseVisitor<
          LeastSquareDiagonalPreconditionerVisitor<Scalar>> {
  using Preconditioner = Eigen::LeastSquareDiagonalPreconditioner<Scalar>;

  template <typename... Ts>
  void execute(nb::class_<Scalar, Ts...>& cl) {
    cl.def(PreconditionerBaseVisitor<Preconditioner>())
        .def("rows", &Preconditioner::rows,
             "Returns the number of rows in the preconditioner.")
        .def("cols", &Preconditioner::rows,
             "Returns the number of cols in the preconditioner.");
  }

  static void expose(nb::module_& m, const char* name) {
    if (check_registration_alias<Preconditioner>(m)) {
      return;
    }
    nb::class_<Preconditioner>(m, name).def(IdVisitor());
  }
};

template <typename Scalar>
void exposeLeastSquareDiagonalPreconditioner(nb::module_& m, const char* name) {
  LeastSquareDiagonalPreconditionerVisitor<Scalar>::expose(m, name);
}
#endif

template <typename Scalar>
struct IdentityPreconditionerVisitor
    : PreconditionerBaseVisitor<Eigen::IdentityPreconditioner> {
  using Preconditioner = Eigen::IdentityPreconditioner;

  template <typename... Ts>
  void execute(nb::class_<Scalar, Ts...>&) {}

  static void expose(nb::module_& m, const char* name) {
    if (check_registration_alias<Preconditioner>(m)) {
      return;
    }
    nb::class_<Preconditioner>(m, name)
        .def(PreconditionerBaseVisitor<Preconditioner>())
        .def(IdVisitor());
  }
};

template <typename Scalar>
void exposeIdentityPreconditioner(nb::module_& m, const char* name) {
  IdentityPreconditionerVisitor<Scalar>::expose(m, name);
}

}  // namespace nanoeigenpy
