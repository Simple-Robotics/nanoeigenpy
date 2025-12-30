# Changelog

All notable changes to this project will be documented in this file.

Change that doesn't affect end user should not be listed:
- CI change
- Github specific file change

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Changed
- Python version update ([#25](https://github.com/Simple-Robotics/nanoeigenpy/pull/25)):
  - Project is now tested with Python 3.10 and 3.14
  - Python 3.10 is the minimal supported Python version

### Added
- Add pixi-build support ([#25](https://github.com/Simple-Robotics/nanoeigenpy/pull/25))

### Removed
- Remove pixi 0.57 warnings ([#25](https://github.com/Simple-Robotics/nanoeigenpy/pull/25))

## [0.4.0] - 2025-07-25

### Added
- Expose additional classes from Eigen (decompositions, solvers, geometry)

## [0.3.0] - 2025-04-28

### Added
- Add equivalent to eigenpy::register_symbolic_link_to_registered_type

## [0.2.1] - 2025-04-25

### Fix
- Fix Windows installation issue

## [0.2.0] - 2025-04-25

### Added
- Add Accelerate support to pixi

## [0.1.0] - 2025-03-31

### Added
- pixi support, CI, and fixup testing
- Bindings for Eigen's decompositions, matrix solvers, and geometry module, and add stubs
- Add BSD-3 Clause license
- Add package.xml

[Unreleased]: https://github.com/Simple-Robotics/nanoeigenpy/compare/v0.4.0...HEAD
[0.3.0]: https://github.com/Simple-Robotics/nanoeigenpy/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/Simple-Robotics/nanoeigenpy/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Simple-Robotics/nanoeigenpy/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Simple-Robotics/nanoeigenpy/releases/tag/v0.1.0
