{ pkgs ? import <nixpkgs> { } }:

# nix provides system tools (uv, ffmpeg).
# uv manages Python itself, driven by `.python-version`.
pkgs.mkShell {
  packages = with pkgs; [
    uv
    ffmpeg
  ];
}
