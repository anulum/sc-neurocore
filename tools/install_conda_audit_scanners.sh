# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Checksum-pinned Linux Conda audit scanners

set -euo pipefail
test "$(uname -s)" = Linux
test "$(uname -m)" = x86_64
audit_scanner_bin="${1:?Provide the scanner binary directory}"
audit_download_dir=$(mktemp -d "${TMPDIR:?Set TMPDIR to an owned directory}/conda-audit-scanners.XXXXXX")
trap 'rm -rf -- "$audit_download_dir"' EXIT
mkdir -p "$audit_scanner_bin"
curl -fsSL --retry 3 \
  https://github.com/anchore/syft/releases/download/v1.55.0/syft_1.55.0_linux_amd64.tar.gz \
  -o "$audit_download_dir/syft.tar.gz"
printf '%s  %s\n' bb65249542db5fedeeb4d3f9b018b75e7264dbfbbb10486f044581a92f90490d \
  "$audit_download_dir/syft.tar.gz" | sha256sum --check --status
tar -xzf "$audit_download_dir/syft.tar.gz" -C "$audit_scanner_bin" syft
curl -fsSL --retry 3 \
  https://github.com/anchore/grype/releases/download/v0.121.0/grype_0.121.0_linux_amd64.tar.gz \
  -o "$audit_download_dir/grype.tar.gz"
printf '%s  %s\n' 64e07133d3486115de647a66581461dccf5c732748a0a3119077395565f15519 \
  "$audit_download_dir/grype.tar.gz" | sha256sum --check --status
tar -xzf "$audit_download_dir/grype.tar.gz" -C "$audit_scanner_bin" grype
