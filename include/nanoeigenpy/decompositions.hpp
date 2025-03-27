/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/decompositions/llt.hpp"
#include "nanoeigenpy/decompositions/ldlt.hpp"
#include "nanoeigenpy/decompositions/minres.hpp"
#include "nanoeigenpy/decompositions/householder-qr.hpp"
#include "nanoeigenpy/decompositions/full-piv-householder-qr.hpp"
#include "nanoeigenpy/decompositions/col-piv-householder-qr.hpp"
#include "nanoeigenpy/decompositions/complete-orthogonal-decomposition.hpp"
#include "nanoeigenpy/decompositions/eigen-solver.hpp"
#include "nanoeigenpy/decompositions/self-adjoint-eigen-solver.hpp"
#include "nanoeigenpy/decompositions/permutation-matrix.hpp"

#include "nanoeigenpy/decompositions/sparse/simplicial-llt.hpp"
#include "nanoeigenpy/decompositions/sparse/simplicial-ldlt.hpp"
#include "nanoeigenpy/decompositions/sparse/cholmod/cholmod-simplicial-llt.hpp"
#include "nanoeigenpy/decompositions/sparse/cholmod/cholmod-simplicial-ldlt.hpp"
#include "nanoeigenpy/decompositions/sparse/cholmod/cholmod-supernodal-llt.hpp"
