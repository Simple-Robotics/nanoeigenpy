#include <nanobind/nanobind.h>

#include "nanoeigenpy/decompositions/ldlt.hpp"
#include "nanoeigenpy/decompositions/llt.hpp"

using namespace nanoeigenpy;

NB_MAKE_OPAQUE(Eigen::LLT<Eigen::MatrixXd>)
NB_MAKE_OPAQUE(Eigen::LDLT<Eigen::MatrixXd>)

NB_MODULE(nanoeigenpy, m) {
  exposeLLTSolver<Eigen::MatrixXd>(m, "LLT");
  exposeLDLTSolver<Eigen::MatrixXd>(m, "LDLT");
}
