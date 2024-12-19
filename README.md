# nanoeigenpy

This is a collection of tools for using Eigen together with nanobind, as a successor to [eigenpy](https://github.com/stack-of-tasks/eigenpy). Its aim is to help the transition away from Boost.Python.

It reintroduces a few features (e.g. bindings for Eigen matrix decompositions) which are not in [nanobind](https://github.com/wjakob/nanobind).

## Rationale

Eigenpy was based on Boost.Python, an aging, complex, heavily templated library with little community support.
Support for many library features initially present in Boost, and which were added to the STL since C++11/14/17, for over a decade (nearly two), were just never added in Boost.Python. This include e.g. support for `{boost,std}::optional`, `{boost,std}::variant`, `{boost,std}::unique_ptr`, proper support for map types... whereas they have been present for years in pybind11 and now nanobind for years.
These features were finally added to eigenpy with a lot of developer effort.

## Installaton

### Dependencies

- the Eigen C++ template library - [conda-forge](https://anaconda.org/conda-forge/eigen) / [GitLab](https://gitlab.com/libeigen/eigen/)
- [nanobind](https://github.com/wjakob/nanobind)
