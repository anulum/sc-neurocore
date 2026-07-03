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

The security-scanner CI packet combines deterministic planning with executable
scanner lanes. The scheduled scanner workflow keeps the heavier fuzz and
benchmark lanes separate from push and pull-request CI. The tag release workflow
uses `tools/security_scan/release_security_sweep.py` to run the release packet,
scanner lanes, repository-owned Semgrep policy, supply-chain audit, bounded
Hypothesis subset, Rust proptest subset, bounded cargo-fuzz subset, and final
artifact index in one sequence.

## What the current workflow generates

- `security_scanner_manifest.json` via `tools/security_scanner_manifest.py`.
- A Python/code plan via `tools/security_scan/python_code_scanner_plan.py`.
- A Rust/supply-chain plan via `tools/security_scan/rust_supply_chain_scanner_plan.py`.
- A model/data licence matrix copy at `security/model_data_license_matrix.json`.
- A release security artifact index from `security/release_artifacts_manifest.json` with
  `tools/security_scan/release_security_artifact_index.py`.
- A release sweep summary at `security/release_security_sweep_summary.json`
  when the tag workflow runs `tools/security_scan/release_security_sweep.py`.
- A Semgrep summary at `security/semgrep_summary.json` when the tag workflow
  runs the release-only `tools/security_scan/run_semgrep_scanners.py` lane
  against `.semgrep.yml`.
- A lightweight scanner lane summary at
  `security/lightweight_scanner_summary.json` when CI runs the executable
  `ruff`, `bandit`, and `actionlint` lane.
- A Rust scanner lane summary at `security/rust_scanner_summary.json` when CI
  runs `cargo-audit` and `cargo-deny`.
- A Python compliance scanner summary at `security/python_compliance_summary.json`
  when CI runs blocking `pip-audit` and non-blocking REUSE lint.
- Machine-readable vulnerability-status fields in the release index:
  `vulnerability_status`, `missing_required_vulnerability_status`,
  `missing_optional_vulnerability_status`, and `vulnerability_summary`.
- Optional scanner artefact slots include `security/ruff.json`,
  `security/bandit.json`, and `security/actionlint.json` as executable
  lightweight scanner outputs.
- Optional Rust scanner artefact slots include `security/cargo_audit.json` and
  `security/cargo_deny.json`; their commands write JSON reports from stdout so
  `cargo audit --file` remains the lockfile input option, not a report path.
- Optional Python compliance artefact slots include `security/pip_audit.json`
  and `security/reuse.json`; `pip-audit` is pinned to an available release and
  runs against the hashed `requirements/release.txt`.
- OSV-Scanner v2 writes `security/osv_scanner.json` and
  `security/osv_scanner_summary.json`; the lane is blocking and runs with the
  pinned Go `1.26.3` toolchain because OSV also evaluates Go standard-library
  vulnerability status from module metadata. The runner scans explicit
  lockfile/requirements inputs with OSV's lockfile plugin rather than recursive
  source discovery so optional development manifests cannot mask the tracked
  dependency surfaces with resolver-side extraction failures. Transient OSV
  resolver service errors are retried before the lane reports failure; any
  remaining validation error or unresolved vulnerability still fails closed.
- Optional typing artefact slots include `security/pyright.json`,
  `security/mypy`, and `security/typing_scanner_summary.json`; the executable
  runner is available for baseline refreshes without enabling the lane in the
  default workflow yet.
- Syft/CycloneDX SBOM generation writes `security/sbom.cdx.json` and validates
  the output with `security/syft_cyclonedx_summary.json`.
- The tag release sweep writes `security/semgrep.json` and
  `security/semgrep_summary.json` from pinned `requirements/semgrep.txt` and
  the repository-owned `.semgrep.yml` policy.
- The tag release sweep writes `security/supply_chain_audit.json` after checking
  the generated SBOM and release requirement hashes with
  `tools/supply_chain_audit.py`.
- The tag release sweep writes `security/hypothesis_fuzz_summary.json` for the
  bounded Python fuzz subset and `security/rust_proptest_summary.json` for the
  Rust proptest subset.
- The nightly/manual cargo-fuzz lane writes `security/cargo_fuzz_summary.json`
  plus per-target reports such as `security/cargo_fuzz_ir_parser.json`; it runs
  outside push and pull-request CI with a bounded total time budget and installs
  the maintained `cargo-fuzz` release pinned in the workflow using stable Cargo
  before executing the fuzz lane on the configured nightly toolchain. The Python
  runner first builds each target with `cargo fuzz build` under an explicit
  build timeout, then runs `cargo fuzz run` with libFuzzer's per-target time
  budget. Per-target reports record whether a failure happened during `build` or
  `run`, including the relevant command tails, so scheduled CI timeouts remain
  actionable instead of being ambiguous scanner failures.
- The nightly/manual benchmark-regression lane writes
  `security/benchmark_regression.json` by regenerating the deterministic
  side-channel benchmark and comparing all numeric metrics against the tracked
  baseline in `benchmarks/baselines/security_side_channel_benchmark.json`.
- The packet summary includes `missing_required_scanner_inputs`; when
  `--fail-on-missing-required` is active, required-input failures inside the
  Python or Rust scanner plans fail the packet even when the packet files
  themselves are present.

## What the packet is and is not

This packet checks availability and planning consistency (manifest shape,
required input paths, and required artifact presence) before release binaries are
launched.

It is therefore a mixed execution/planning envelope:

- lightweight scanner commands are executed for `ruff`, `bandit`, and
  `actionlint`,
- Rust scanner commands are executed for `cargo-audit` and `cargo-deny`,
- Python compliance commands are executed for blocking `pip-audit` and
  non-blocking REUSE lint,
- OSV-Scanner v2 is executed in the main packet lane and fails closed on
  unresolved vulnerabilities or invalid/missing JSON output,
- typing commands for Pyright and strict Mypy have an executable runner and
  release-packet artefact slots, but remain outside the default workflow until
  the repo-wide type baseline is triaged,
- Syft/CycloneDX SBOM generation is executed in the main packet lane and fails
  closed if the SBOM is missing or not a CycloneDX JSON document,
- Semgrep is executed by the tag release sweep from `.semgrep.yml` and fails
  closed on findings because the lane uses `--error`,
- cargo-fuzz commands are executed by the separate scheduled/manual
  `nightly-cargo-fuzz` workflow job and by the tag release sweep when
  `--include-fuzz` is set,
- bounded Hypothesis and Rust proptest subsets are executed by the tag release
  sweep,
- benchmark-regression commands are executed by the separate scheduled/manual
  `nightly-benchmark-regression` workflow job,
- heavier scanner commands such as Trivy FS and Gitleaks remain represented as
  manifest entries until their pinned install lanes are promoted,
  and
- no direct `trivy fs`, `cargo-fuzz`, `gitleaks`, or similar heavyweight
  commands are executed in push or pull-request CI.

After the lightweight lane runs, the workflow regenerates
`release_security_artifact_index.json` against `security/ci-security-packet` so
the uploaded index reflects the scanner artefacts that were actually produced.

## Relevant CLI references

- `python tools/security_scanner_manifest.py --output security/security_scanner_manifest.json`
- `python tools/security_scanner_manifest.py --validate security/security_scanner_manifest.json`
- `python tools/security_scan/ci_security_packet.py --output-dir security/ci-security-packet --fail-on-missing-required`
- `python tools/security_scan/run_lightweight_security_scanners.py --output-dir security/ci-security-packet`
- `python tools/security_scan/run_rust_security_scanners.py --output-dir security/ci-security-packet`
- `python tools/security_scan/run_python_compliance_scanners.py --output-dir security/ci-security-packet`
- `python tools/security_scan/run_osv_scanners.py --output-dir security/ci-security-packet`
- `python tools/security_scan/run_typing_scanners.py --output-dir security/ci-security-packet`
- `python tools/security_scan/run_syft_cyclonedx_scanners.py --output-dir security/ci-security-packet`
- `python tools/security_scan/run_semgrep_scanners.py --output-dir security/ci-security-packet`
- `python tools/security_scan/release_security_sweep.py --output-dir security/ci-security-packet --include-fuzz --fuzz-max-total-time 300`
- `python tools/security_scan/run_cargo_fuzz_scanners.py --output-dir security/cargo-fuzz-packet --target all --max-total-time 300 --build-timeout 900`
- `python tools/security_scan/run_benchmark_regression_scanners.py --baseline benchmarks/baselines/security_side_channel_benchmark.json --current security/benchmark-current/security_side_channel_benchmark.json --output security/benchmark-regression-packet/security/benchmark_regression.json --max-regression-pct 5.0`
- `python tools/security_scan/python_code_scanner_plan.py`
- `python tools/security_scan/rust_supply_chain_scanner_plan.py`
- `python tools/security_scan/release_security_artifact_index.py --manifest security/release_artifacts_manifest.json --root . --output security/release_security_artifact_index.json`

The packet is used as a compliance aid for security and release workflows before
heavyweight scanners are enabled in the normal runtime chain. Both the security
packet workflow and tagged release workflow use `--fail-on-missing-required` so
missing required packet artefacts fail closed.

REUSE lint is intentionally non-blocking in this lane while the repository-wide
legacy SPDX coverage debt is remediated. The JSON report is still uploaded in
the packet so the remaining file-level compliance gap is visible and measurable.
The only OSV exception is a bounded `RUSTSEC-2024-0436` entry for the transitive
`wgpu -> metal -> paste` path; `cargo audit` classifies that item as an
unmaintained warning, and the exception expires on 2026-08-31 unless the GPU
backend dependency path is upgraded or replaced earlier.
On tagged releases, `.github/workflows/release.yml` runs the release security
sweep and attaches both
`security/ci-security-packet/release_security_artifact_index.json` and
`security/ci-security-packet/security/release_security_sweep_summary.json` to
the GitHub Release assets.
