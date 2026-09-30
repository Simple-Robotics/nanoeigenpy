#! /bin/bash
# Activation script

if [[ $PIXI_ENVIRONMENT_PLATFORMS == *"linux"* ]];
then
  # Conda compiler is named x86_64-conda-linux-gnu-c++, ccache can't resolve it
  # (https://ccache.dev/manual/latest.html#config_compiler_type)
  export CCACHE_COMPILERTYPE=gcc
fi

# Without -isystem, some LSP can't find headers
export NANOEIGENPY_CXX_FLAGS="$NANOEIGENPY_CXX_FLAGS -isystem $CONDA_PREFIX/include"

# Set default build value only if not previously set
export NANOEIGENPY_BUILD_TYPE=${NANOEIGENPY_BUILD_TYPE:=Release}
export NANOEIGENPY_CHOLMOD_SUPPORT=${NANOEIGENPY_CHOLMOD_SUPPORT:=OFF}
export NANOEIGENPY_ACCELERATE_SUPPORT=${NANOEIGENPY_ACCELERATE_SUPPORT:=OFF}
