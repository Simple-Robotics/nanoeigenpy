/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include "nanoeigenpy/eigen-base.hpp"
#include <nanobind/operators.h>

namespace nanoeigenpy {
namespace nb = nanobind;
using namespace nb::literals;

template <int SizeAtCompileTime, int MaxSizeAtCompileTime = SizeAtCompileTime,
          typename StorageIndex_ = int>
void exposePermutationMatrix(nb::module_ m, const char *name) {
  using StorageIndex = StorageIndex_;
  using PermutationMatrix =
      Eigen::PermutationMatrix<SizeAtCompileTime, MaxSizeAtCompileTime,
                               StorageIndex>;
  using VectorIndex = Eigen::Matrix<StorageIndex, SizeAtCompileTime, 1, 0,
                                    MaxSizeAtCompileTime, 1>;

  if (check_registration_alias<PermutationMatrix>(m)) {
    return;
  }
  nb::class_<PermutationMatrix>(m, name,
                                "Permutation matrix.\n"
                                "This class represents a permutation matrix, "
                                "internally stored as a vector of integers.")

      .def(nb::init<Eigen::DenseIndex>(), "size"_a,
           "Default constructor with memory preallocation.")
      .def(nb::init<VectorIndex>(), "indices"_a,
           "The indices array has the meaning that the permutations sends "
           "each integer i to indices[i].\n"
           "It is your responsibility to check that the indices array that "
           "you passes actually describes a permutation, i.e., each value "
           "between 0 and n-1 occurs exactly once, where n is the array's "
           "size.")

      .def(
          "indices",
          [](const PermutationMatrix &self) {
            return VectorIndex(self.indices());
          },
          "The stored array representing the permutation.")

      .def("applyTranspositionOnTheLeft",
           &PermutationMatrix::applyTranspositionOnTheLeft, "i"_a, "j"_a,
           "Multiplies self by the transposition (ij) on the left.")
      .def("applyTranspositionOnTheRight",
           &PermutationMatrix::applyTranspositionOnTheRight, "i"_a, "j"_a,
           "Multiplies self by the transposition (ij) on the right.")

      .def(
          "setIdentity", [](PermutationMatrix &self) { self.setIdentity(); },
          "Sets self to be the identity permutation matrix.")
      .def(
          "setIdentity",
          [](PermutationMatrix &self, Eigen::DenseIndex size) {
            self.setIdentity(size);
          },
          "size"_a,
          "Sets self to be the identity permutation matrix of given size.")

      .def("toDenseMatrix", &PermutationMatrix::toDenseMatrix,
           "Returns a numpy array object initialized from this permutation "
           "matrix.")

      .def(
          "transpose",
          [](const PermutationMatrix &self) -> PermutationMatrix {
            return self.transpose();
          },
          "Returns the tranpose permutation matrix.")
      .def(
          "inverse",
          [](const PermutationMatrix &self) -> PermutationMatrix {
            return self.inverse();
          },
          "Returns the inverse permutation matrix.")

      .def("resize", &PermutationMatrix::resize, "size"_a,
           "Resizes to given size.")

      .def(nb::self * nb::self)
      .def(EigenBaseVisitor())

      .def(IdVisitor());
}

}  // namespace nanoeigenpy
