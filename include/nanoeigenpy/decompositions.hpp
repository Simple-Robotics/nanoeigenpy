/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/decompositions/llt.hpp"
#include "nanoeigenpy/decompositions/ldlt.hpp"
#include "nanoeigenpy/decompositions/householder-qr.hpp"
#include "nanoeigenpy/decompositions/full-piv-householder-qr.hpp"
#include "nanoeigenpy/decompositions/col-piv-householder-qr.hpp"
#include "nanoeigenpy/decompositions/complete-orthogonal-decomposition.hpp"
#include "nanoeigenpy/decompositions/eigen-solver.hpp"
#include "nanoeigenpy/decompositions/self-adjoint-eigen-solver.hpp"
#include "nanoeigenpy/decompositions/generalized-self-adjoint-eigen-solver.hpp"
#include "nanoeigenpy/decompositions/complex-eigen-solver.hpp"
#include "nanoeigenpy/decompositions/complex-schur.hpp"
#include "nanoeigenpy/decompositions/generalized-eigen-solver.hpp"
#include "nanoeigenpy/decompositions/hessenberg-decomposition.hpp"
#include "nanoeigenpy/decompositions/real-qz.hpp"
#include "nanoeigenpy/decompositions/real-schur.hpp"
#include "nanoeigenpy/decompositions/tridiagonalization.hpp"
#include "nanoeigenpy/decompositions/permutation-matrix.hpp"
#include "nanoeigenpy/decompositions/full-piv-lu.hpp"
#include "nanoeigenpy/decompositions/partial-piv-lu.hpp"
#include "nanoeigenpy/decompositions/bdcsvd.hpp"
#include "nanoeigenpy/decompositions/jacobi_svd.hpp"

#include "nanoeigenpy/decompositions/sparse/simplicial-llt.hpp"
#include "nanoeigenpy/decompositions/sparse/simplicial-ldlt.hpp"
#include "nanoeigenpy/decompositions/sparse/sparse-lu.hpp"
#include "nanoeigenpy/decompositions/sparse/sparse-qr.hpp"

#ifdef NANOEIGENPY_HAS_CHOLMOD
#include "nanoeigenpy/decompositions/sparse/cholmod/cholmod-simplicial-llt.hpp"
#include "nanoeigenpy/decompositions/sparse/cholmod/cholmod-simplicial-ldlt.hpp"
#include "nanoeigenpy/decompositions/sparse/cholmod/cholmod-supernodal-llt.hpp"
#endif
#ifdef NANOEIGENPY_HAS_ACCELERATE
#include "nanoeigenpy/decompositions/sparse/accelerate/accelerate.hpp"
#endif
