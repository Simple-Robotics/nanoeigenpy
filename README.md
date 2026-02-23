# nanoeigenpy

<p align="left">
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Linter: ruff"></a>
  <a href="https://anaconda.org/conda-forge/nanoeigenpy"><img src="https://img.shields.io/conda/vn/conda-forge/nanoeigenpy.svg"></a>
</p>

This is a collection of tools for using Eigen together with nanobind, as a successor to the [eigenpy](https://github.com/stack-of-tasks/eigenpy) support library. Its aim is to help the transition away from Boost.Python.

It reintroduces a few features (e.g. bindings for Eigen matrix decompositions) which are not in [nanobind](https://github.com/wjakob/nanobind) at time of writing.

## Rationale

Eigenpy was based on Boost.Python, an aging, complex, heavily templated library with little community support.
Support for many library features initially present in Boost, and which were added to the STL since C++11/14/17, for over a decade (nearly two), were just never added in Boost.Python. This includes support for `{boost,std}::optional`, `{boost,std}::variant`, `{boost,std}::unique_ptr`, proper support for map types... whereas they have been present in pybind11, and now nanobind, for years.

These features were finally added to eigenpy with a lot of developer effort. This created additional need for supporting these additional features ourselves, including many downstream consumers (mainly in the robotics community).

## Features

**nanoeigenpy** provides the following features to help you write bindings between Eigen and Python:

- bindings for Eigen's [Geometry module](https://libeigen.gitlab.io/docs/group__Geometry__Module.html) - quaternions, angle-axis representations...
- bindings for Eigen's matrix dense and sparse decompositions and solvers

### Optional features

**nanoeigenpy** also provides bindings for Eigen's [Cholmod](https://eigen.tuxfamily.org/dox/group__CholmodSupport__Module.html) and [Apple Accelerate](https://eigen.tuxfamily.org/dox/group__AccelerateSupport__Module.html) modules.

> [!NOTE]
> The Accelerate module is available since Eigen 5.0 (Oct. 2025).

Cholmod is part of the [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) algorithms library. It can be installed standalone from [conda](https://anaconda.org/conda-forge/libcholmod).


## Example usage

The features included in **nanoeigenpy** are distributed in a Python module which can be imported, or through standalone headers which can be included in your own Python bindings code using a CMake target.

### Using the nanoeigenpy headers (with CMake)

To directly use the tools in **nanoeigenpy**'s headers, link to it in CMake (or whichever build tool you have, but only CMake support is planned so far).

```cmake
# look for the nanoeigenpy CMake package
find_package(nanoeigenpy REQUIRED)

nanobind_add_module(my_ext NB_STATIC my_ext.cpp)
target_link_libraries(my_ext PRIVATE nanoeigenpy::nanoeigenpy_headers)
```

Then, in your C++ extension module code, include the relevant headers and call functions to expose the required type:

```cpp
#include <nanoeigenpy/geometry/quaternion.hpp>

namespace nb = nanobind;

void f(const Eigen::Quaterniond &quat) {
    // ...
}

NB_MODULE(my_ext, m) {
    nanoeigenpy::exposeQuaternion<double>(m, "Quaternion");
    m.def("f", f, nb::arg("quat"));
}
```

### Using the compiled Python module

In the case above, **nanoeigenpy**'s Python extension module already includes bindings for `Eigen::Quaternion` with the `double` scalar type (AKA `Eigen::Quaterniond`). Then, we can simply get nanobind to import it in our extension module:

```cpp
#include <Eigen/Geometry>

namespace nb = nanobind;

void f(const Eigen::Quaterniond &quat) {
    // ...
}

NB_MODULE(my_ext, m) {
    // import nanoeigenpy's module **here**
    nb::module_::import_("nanoeigenpy");
    m.def("f", f, nb::arg("quat"));
}
```

Alternatively, Python code which uses our extension `my_ext` can also bring in **nanoeigenpy**:

```python
import nanoeigenpy
from nanoeigenpy import Quaternion
from my_ext import f

quat = Quaternion(0., 1., 0., 0.)
f(quat)
```

> [!NOTE]
> If you have a specific scalar type (e.g. `float16`) with which you want to use `Eigen::Quaternion`, or matrix solvers, or other features in **nanoeigenpy**, you should refer to the first approach and use **nanoeigenpy** from C++ directly.

Furthermore, you can check the available SIMD instruction sets from the Python extension module itself:

```python
>>> import nanoeigenpy
>>> print(nanoeigenpy.SimdInstructionSetsInUse())
>>> SSE, SSE2  # no optimizations
```


## Installation

### Dependencies

- the Eigen C++ template library - [conda-forge](https://anaconda.org/conda-forge/eigen) | [repo](https://gitlab.com/libeigen/eigen/)
- nanobind - [conda-forge](https://anaconda.org/conda-forge/nanobind) | [repo](https://github.com/wjakob/nanobind)
- [for testing] pytest - `conda install pytest` or `pip install pytest`

#### Conda

```bash
conda install -c conda-forge nanobind eigen  # or mamba install
```

#### Building

```bash
cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX=<your-prefix>  # prefix can be e.g. $CONDA_PREFIX
cd build/
cmake --build . --target install
```

## Credits

The following people have been involved in the development of **nanoeigenpy**:

- [Wilson Jallet](https://manifoldfr.github.io/) (Inria): core developer and manager of the project
- [Lucas Haubert](https://www.linkedin.com/in/lucas-haubert-b668a421a/) (Inria): core developer
- [Justin Carpentier](https://jcarpent.github.io) (Inria): core developer
- [Joris Vailant](https://github.com/jorisv) (Inria): windows support

If you have taken part in the development of **nanoeigenpy**, feel free to add your name and contribution here.
