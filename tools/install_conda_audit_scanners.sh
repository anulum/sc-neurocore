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
  https://github.com/anchore/syft/releases/download/v1.54.1/syft_1.54.1_linux_amd64.tar.gz \
  -o "$audit_download_dir/syft.tar.gz"
printf '%s  %s\n' c069905b391cc4c20a5ba65ad5c10be2a7ba074f8ea6ad203e24d14e303dad47 \
  "$audit_download_dir/syft.tar.gz" | sha256sum --check --status
tar -xzf "$audit_download_dir/syft.tar.gz" -C "$audit_scanner_bin" syft
curl -fsSL --retry 3 \
  https://github.com/anchore/grype/releases/download/v0.120.1/grype_0.120.1_linux_amd64.tar.gz \
  -o "$audit_download_dir/grype.tar.gz"
printf '%s  %s\n' 0a9ee97ef5ae2ee953b0a80098105052e846cdbe319a57d808b519c33cd1343d \
  "$audit_download_dir/grype.tar.gz" | sha256sum --check --status
tar -xzf "$audit_download_dir/grype.tar.gz" -C "$audit_scanner_bin" grype
