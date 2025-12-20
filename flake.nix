{
  description = "Bindings between Numpy and Eigen using nanobind";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    # Test https://github.com/jrl-umi3218/jrl-cmakemodules/pull/798
    jrl-cmakemodules = {
      url = "github:ahoarau/jrl-cmakemodules/jrl-next";
      inputs.flake-parts.follows = "flake-parts";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } (
      { self, lib, ... }:
      {
        systems = inputs.nixpkgs.lib.systems.flakeExposed;
        flake.overlays = {
          default = final: prev: {
            pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
              (python-final: python-prev: {
                nanoeigenpy = python-prev.nanoeigenpy.overrideAttrs (old: {
                  cmakeFlags = (old.cmakeFlags or [ ]) ++ [
                    "-DBUILD_TESTING=ON"
                    "-DBUILD_WITH_CHOLMOD_SUPPORT=OFF"
                  ];
                  nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
                    python-final.pytest
                  ];
                  # Don’t produce/require a separate doc output
                  outputs = [ "out" ];
                  postPatch = "";
                  postFixup = "";
                  src = lib.fileset.toSource {
                    root = ./.;
                    fileset = lib.fileset.unions [
                      ./cmake
                      ./CMakeLists.txt
                      ./include
                      ./package.xml
                      ./src
                      ./tests
                    ];
                  };
                });
              })
            ];
          };
        };
        perSystem =
          {
            pkgs,
            self',
            system,
            ...
          }:
          {
            _module.args = {
              pkgs = import inputs.nixpkgs {
                inherit system;
                overlays = [
                  inputs.jrl-cmakemodules.overlays.default
                  self.overlays.default
                ];
              };
            };
            apps.default = {
              type = "app";
              program = pkgs.python3.withPackages (_: [ self'.packages.default ]);
            };
            packages = {
              default = self'.packages.nanoeigenpy;
              jrl-cmakemodules = pkgs.jrl-cmakemodules;
              nanoeigenpy = pkgs.python3Packages.nanoeigenpy;
            };
          };
      }
    );
}
