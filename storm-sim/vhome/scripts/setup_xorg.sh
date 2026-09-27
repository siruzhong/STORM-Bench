#!/usr/bin/env bash
# Extract a private Xorg runtime without installing system packages.
set -euo pipefail
root="${1:-.runtime/xorg}"
mkdir -p "$root"
cd "$root"
apt-get download xserver-xorg-core libxcvt0 libxfont2 libfontenc1
for package in ./*.deb; do
    dpkg-deb -x "$package" .
done
printf 'Xorg runtime: %s\n' "$PWD"
