/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/eigen-base.hpp"
#include <nanobind/operators.h>

namespace nanoeigenpy {
namespace nb = nanobind;

template <int SizeAtCompileTime, int MaxSizeAtCompileTime = SizeAtCompileTime,
          typename StorageIndex_ = int>
void exposePermutationMatrix(nb::module_ m, const char *name) {
  using StorageIndex = StorageIndex_;
  using PermutationMatrix =
      Eigen::PermutationMatrix<SizeAtCompileTime, MaxSizeAtCompileTime,
                               StorageIndex>;
  using VectorIndex = Eigen::Matrix<StorageIndex, SizeAtCompileTime, 1, 0,
                                    MaxSizeAtCompileTime, 1>;

  auto cl =
      nb::class_<PermutationMatrix>(
          m, name,
          "Permutation matrix.\n"
          "This class represents a permutation matrix, "
          "internally stored as a vector of integers.")

          .def(nb::init<Eigen::DenseIndex>(), nb::arg("size"),
               "Default constructor with memory preallocation.")
          .def(nb::init<VectorIndex>(), nb::arg("indices"),
               "The indices array has the meaning that the permutations sends "
               "each integer i to indices[i].\n"
               "It is your responsibility to check that the indices array that "
               "you passes actually describes a permutation, i.e., each value "
               "between 0 and n-1 occurs exactly once, where n is the array's "
               "size.")

          .def(
              "indices",
              [](const PermutationMatrix &self) -> const auto & {
                return VectorIndex(self.indices());
              },
              "The stored array representing the permutation.")

          .def("applyTranspositionOnTheLeft",
               &PermutationMatrix::applyTranspositionOnTheLeft, nb::arg("i"),
               nb::arg("j"),
               "Multiplies self by the transposition (ij) on the left.")
          .def("applyTranspositionOnTheRight",
               &PermutationMatrix::applyTranspositionOnTheRight, nb::arg("i"),
               nb::arg("j"),
               "Multiplies self by the transposition (ij) on the right.")

          .def(
              "setThreshold",
              [](PermutationMatrix &self) { self.setIdentity(); },
              "Sets self to be the identity permutation matrix.")
          .def(
              "setThreshold",
              [](PermutationMatrix &self, Eigen::DenseIndex size) {
                self.setIdentity(size);
              },
              nb::arg("size"),
              "Sets self to be the identity permutation matrix of given size.")

          .def("toDenseMatrix", &PermutationMatrix::toDenseMatrix,
               "Returns a numpy array object initialized from this permutation "
               "matrix.")

          .def(
              "transpose",
              [](PermutationMatrix const &self) -> PermutationMatrix {
                return self.transpose();
              },
              "Returns the tranpose permutation matrix.")
          .def(
              "inverse",
              [](PermutationMatrix const &self) -> PermutationMatrix {
                return self.inverse();
              },
              "Returns the inverse permutation matrix.")

          .def("resize", &PermutationMatrix::resize, nb::arg("size"),
               "Resizes to given size.")

          .def(nb::self * nb::self)
          .def(EigenBaseVisitor())

          .def(
              "id",
              [](PermutationMatrix const &self) -> int64_t {
                return reinterpret_cast<int64_t>(&self);
              },
              "Returns the unique identity of an object.\n"
              "For objects held in C++, it corresponds to its memory address.");
}

}  // namespace nanoeigenpy

// TODO

// Manage the _ in the typenames, why and how ?
