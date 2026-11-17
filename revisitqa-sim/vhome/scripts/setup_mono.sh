#!/usr/bin/env bash
# Extract the compiler and assembly tools without installing system packages.
set -euo pipefail
root="${1:-.runtime/mono}"
mkdir -p "$root"
cd "$root"
apt-get download mono-mcs mono-runtime-sgen mono-runtime-common mono-gac \
    libmono-corlib4.5-cil libmono-corlib4.5-dll libmono-system-core4.0-cil \
    libmono-system-xml4.0-cil libmono-system4.0-cil libmono-microsoft-csharp4.0-cil \
    libmono-security4.0-cil libmono-cecil-private-cil
for package in ./*.deb; do
    dpkg-deb -x "$package" .
done
