/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/solvers/basic-preconditioners.hpp"
#include "nanoeigenpy/solvers/bfgs-preconditioners.hpp"
#include "nanoeigenpy/solvers/iterative-solver-base.hpp"
#include "nanoeigenpy/solvers/minres.hpp"
#if EIGEN_VERSION_AT_LEAST(3, 3, 5)
#include "nanoeigenpy/solvers/least-squares-conjugate-gradient.hpp"
#endif
#include "nanoeigenpy/solvers/conjugate-gradient.hpp"

#include "nanoeigenpy/solvers/bicgstab.hpp"
