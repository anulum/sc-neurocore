<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SC-NeuroCore - Security scanner CI packet
-->

# Security Scanner CI Packet

The current security-scanner CI packet is an offline dry-run planning layer.
It does not execute heavyweight scanner binaries in CI yet.

## What the current workflow generates

- `security_scanner_manifest.json` via `tools/security_scanner_manifest.py`.
- A Python/code plan via `tools/security_scan/python_code_scanner_plan.py`.
- A Rust/supply-chain plan via `tools/security_scan/rust_supply_chain_scanner_plan.py`.
- A model/data licence matrix copy at `security/model_data_license_matrix.json`.
- A release security artifact index from `security/release_artifacts_manifest.json` with
  `tools/security_scan/release_security_artifact_index.py`.

## What the packet is and is not

This packet checks availability and planning consistency (manifest shape, required input
paths, and required artifact presence) before binaries are launched.

It is therefore a planning envelope only:

- scanner commands are represented as entries in the plan,
- heavy binaries are intentionally deferred, and
- no direct `trivy fs`, `osv-scanner`, `cargo-fuzz`, `gitleaks`, or similar heavyweight
  commands are executed in this stage.

## Relevant CLI references

- `python tools/security_scanner_manifest.py --output security/security_scanner_manifest.json`
- `python tools/security_scanner_manifest.py --validate security/security_scanner_manifest.json`
- `python tools/security_scan/ci_security_packet.py --output-dir security/ci-security-packet`
- `python tools/security_scan/python_code_scanner_plan.py`
- `python tools/security_scan/rust_supply_chain_scanner_plan.py`
- `python tools/security_scan/release_security_artifact_index.py --manifest security/release_artifacts_manifest.json --root . --output security/release_security_artifact_index.json`

The packet is used as a compliance aid for security and release workflows before
heavyweight scanners are enabled in the normal runtime chain.
