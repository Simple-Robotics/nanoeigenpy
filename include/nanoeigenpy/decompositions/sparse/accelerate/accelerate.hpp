/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include "nanoeigenpy/id.hpp"
#include "nanoeigenpy/decompositions/sparse/sparse-solver-base.hpp"

#include <Eigen/AccelerateSupport>

namespace nanoeigenpy {

template <typename AccelerateDerived>
struct AccelerateImplVisitor
    : nb::def_visitor<AccelerateImplVisitor<AccelerateDerived>> {
  using Solver = AccelerateDerived;
  using MatrixType = typename AccelerateDerived::MatrixType;
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  using CholMatrixType = MatrixType;
  using StorageIndex = typename MatrixType::StorageIndex;

  template <typename... Ts>
  void execute(nb::class_<Solver, Ts...>& cl) {
    using namespace nb::literals;

    cl.def(nb::init<>(), "Default constructor.")
        .def(nb::init<const MatrixType&>(), "matrix"_a,
             "Initialize the solver with matrix A for further Ax=b solving.\n"
             "This constructor is a shortcut for the default constructor "
             "followed by a call to compute().")

        .def("analyzePattern", &Solver::analyzePattern,
             "Performs a symbolic decomposition on the sparcity of matrix.\n"
             "This function is particularly useful when solving for several "
             "problems having the same structure.")

        .def(SparseSolverBaseVisitor())

        .def(
            "compute",
            [](Solver& c, const MatrixType& matrix) {
              return c.compute(matrix);
            },
            "matrix"_a,
            "Computes the sparse Cholesky decomposition of a given matrix.",
            nb::rv_policy::reference)

        .def("factorize", &Solver::factorize, "matrix"_a,
             "Performs a numeric decomposition of a given matrix.\n"
             "The given matrix must has the same sparcity than the matrix on "
             "which the symbolic decomposition has been performed.\n"
             "See also analyzePattern().")

        .def("info", &Solver::info,
             "NumericalIssue if the input contains INF or NaN values or "
             "overflow occured. Returns Success otherwise.")

        .def("setOrder", &Solver::setOrder, "Set order");
  }

  static void expose(nb::module_& m, const char* name, const char* doc) {
    if (check_registration_alias<Solver>(m)) {
      return;
    }
    nb::class_<Solver>(m, name, doc)
        .def(AccelerateImplVisitor())
        .def(IdVisitor());
  }
};

inline void exposeAccelerate(nb::module_ m) {
  using namespace Eigen;
  using ColMajorSparseMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor>;

#define EXPOSE_ACCELERATE_DECOMPOSITION(m, name, doc) \
  AccelerateImplVisitor<name<ColMajorSparseMatrix>>::expose(m, #name, doc)

  EXPOSE_ACCELERATE_DECOMPOSITION(
      m, AccelerateLLT,
      "A direct Cholesky (LLT) factorization and solver based on Accelerate.");
  EXPOSE_ACCELERATE_DECOMPOSITION(m, AccelerateLDLT,
                                  "The default Cholesky (LDLT) factorization "
                                  "and solver based on Accelerate.");
  EXPOSE_ACCELERATE_DECOMPOSITION(
      m, AccelerateLDLTUnpivoted,
      "A direct Cholesky-like LDL^T factorization and solver based on "
      "Accelerate with only 1x1 pivots and no pivoting.");
  EXPOSE_ACCELERATE_DECOMPOSITION(
      m, AccelerateLDLTSBK,
      "A direct Cholesky (LDLT) factorization and solver based on Accelerate "
      "with Supernode Bunch-Kaufman and static pivoting.");
  EXPOSE_ACCELERATE_DECOMPOSITION(
      m, AccelerateLDLTTPP,
      "A direct Cholesky (LDLT) factorization and solver based on Accelerate "
      "with full threshold partial pivoting.");
  EXPOSE_ACCELERATE_DECOMPOSITION(
      m, AccelerateQR, "A QR factorization and solver based on Accelerate.");
  EXPOSE_ACCELERATE_DECOMPOSITION(
      m, AccelerateCholeskyAtA,
      "A QR factorization and solver based on Accelerate without storing Q "
      "(equivalent to A^TA = R^T R).");
#undef EXPOSE_ACCELERATE_DECOMPOSITION
}

}  // namespace nanoeigenpy
