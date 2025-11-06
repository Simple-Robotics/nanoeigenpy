#! /bin/bash
# Activation script

export CC="clang"
export CXX="clang++"

# Force the use of lld because LLVMgold is not packaged.
# See https://github.com/conda-forge/llvmdev-feedstock/issues/172
# LLVMgold is used because nanoeigenpy activate CMAKE_INTERPROCEDURAL_OPTIMIZATION.
export NANOEIGENPY_CXX_FLAGS="$NANOEIGENPY_CXX_FLAGS -fuse-ld=lld"
