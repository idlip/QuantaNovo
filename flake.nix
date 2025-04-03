{
  description = "Example Nix development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/2795c506fe8fb7b03c36ccb51f75b6df0ab2553f";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            git uv
          ];
          shellHook = ''
            echo "Welcome to QuantaNet algorithm development.!"
          '';
        };
      }
    );
}
