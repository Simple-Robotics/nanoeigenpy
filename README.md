# nanoeigenpy

This is a collection of tools for using Eigen together with nanobind, as a successor to the [eigenpy](https://github.com/stack-of-tasks/eigenpy) support library. Its aim is to help the transition away from Boost.Python.

It reintroduces a few features (e.g. bindings for Eigen matrix decompositions) which are not in [nanobind](https://github.com/wjakob/nanobind) at time of writing.

## Rationale

Eigenpy was based on Boost.Python, an aging, complex, heavily templated library with little community support.
Support for many library features initially present in Boost, and which were added to the STL since C++11/14/17, for over a decade (nearly two), were just never added in Boost.Python. This includes support for `{boost,std}::optional`, `{boost,std}::variant`, `{boost,std}::unique_ptr`, proper support for map types... whereas they have been present in pybind11, and now nanobind, for years.

These features were finally added to eigenpy with a lot of developer effort. This created additional need for supporting these additional features ourselves, including many downstream consumers (mainly in the robotics community).

## Features

- bindings for Eigen's [Geometry module](https://libeigen.gitlab.io/docs/group__Geometry__Module.html) - quaternions, angle-axis representations...
- bindings for Eigen's matrix dense and sparse decompositions and solvers

## Installation

### Dependencies

- the Eigen C++ template library - [conda-forge](https://anaconda.org/conda-forge/eigen) | [repo](https://gitlab.com/libeigen/eigen/)
- nanobind - [conda-forge](https://anaconda.org/conda-forge/nanobind) | [repo](https://github.com/wjakob/nanobind)

#### Conda

```bash
conda install -c conda-forge nanobind eigen  # or mamba install
```
