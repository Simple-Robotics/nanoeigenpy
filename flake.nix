{
  description = "Bindings between Numpy and Eigen using nanobind";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = inputs.nixpkgs.lib.systems.flakeExposed;
      perSystem =
        { pkgs, self', ... }:
        {
          packages = {
            default = self'.packages.nanoeigenpy;
            eigen = pkgs.eigen.overrideAttrs {
              # Apply https://gitlab.com/libeigen/eigen/-/merge_requests/977
              postPatch = ''
                substituteInPlace Eigen/src/SVD/BDCSVD.h \
                  --replace-fail "if (l == 0) {" "if (i >= k && l == 0) {"
              '';
            };
            nanoeigenpy =
              (pkgs.python3Packages.nanoeigenpy.override { inherit (self'.packages) eigen; }).overrideAttrs
                (_: {
                  src = pkgs.lib.fileset.toSource {
                    root = ./.;
                    fileset = pkgs.lib.fileset.unions [
                      ./CMakeLists.txt
                      ./include
                      ./package.xml
                      ./src
                      ./tests
                    ];
                  };
                });
          };
        };
    };
}
