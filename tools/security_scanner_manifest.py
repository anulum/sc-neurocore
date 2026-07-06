#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Define a deterministic security scanner manifest used by CI wiring.

This module is pure configuration: it describes pinned scanner commands,
ownership, cadence, and blocking/noise policy. It is intentionally offline
and does not execute scanners.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCAN_MANIFEST_SCHEMA_VERSION = "sc-neurocore.security-scanner-manifest.v1"


@dataclass(frozen=True)
class ScannerInput:
    path: str
    purpose: str
    required: bool


@dataclass(frozen=True)
class ScannerManifestEntry:
    name: str
    ecosystem: str
    cadence: str
    blocking_policy: str
    command: str
    inputs: tuple[ScannerInput, ...]
    owner: str
    noise: str
    pinned_version: str
    allowed_to_fail_rationale: str | None


def _input(path: str, purpose: str, *, required: bool = True) -> ScannerInput:
    return ScannerInput(path=path, purpose=purpose, required=required)


_MANDATORY_SCANNERS = (
    ScannerManifestEntry(
        name="pip-audit",
        ecosystem="python",
        cadence="on-push",
        blocking_policy="blocking",
        command=(
            "pip-audit --strict --requirement requirements/release.txt "
            "--format json --progress-spinner off --output security/pip_audit.json"
        ),
        inputs=(
            _input(
                path="requirements/release.txt",
                purpose="Pinned dependencies and hashes for supply-chain audit",
            ),
            _input(path="pyproject.toml", purpose="Dependency metadata reference", required=False),
        ),
        owner="SC-NeuroCore security lane owner",
        noise="low",
        pinned_version="pip-audit==2.10.0",
        allowed_to_fail_rationale=None,
    ),
    ScannerManifestEntry(
        name="osv-scanner",
        ecosystem="multi",
        cadence="on-push",
        blocking_policy="blocking",
        command=(
            "python tools/security_scan/run_osv_scanners.py "
            "--output-dir security/ci-security-packet"
        ),
        inputs=(
            _input(
                path="tools/security_scan/osv-scanner.toml",
                purpose="OSV-Scanner v2 exception and override policy",
            ),
            _input(
                path="requirements/release.txt",
                purpose="Python dependencies for vulnerability correlation",
            ),
            _input(path="Cargo.lock", purpose="Rust dependency lockfile", required=False),
        ),
        owner="SC-NeuroCore security lane owner",
        noise="low",
        pinned_version="osv-scanner==2.3.8",
        allowed_to_fail_rationale=None,
    ),
    ScannerManifestEntry(
        name="cargo-audit",
        ecosystem="rust",
        cadence="on-push",
        blocking_policy="blocking",
        command="cargo audit --format json --file Cargo.lock",
        inputs=(
            _input(
                path="Cargo.lock",
                purpose="Cargo dependency lockfile and advisory matching",
            ),
            _input(path="Cargo.toml", purpose="Workspace manifest reference"),
        ),
        owner="Rust security maintainer",
        noise="low",
        pinned_version="cargo-audit==0.22.1",
        allowed_to_fail_rationale=None,
    ),
    ScannerManifestEntry(
        name="cargo-deny",
        ecosystem="rust",
        cadence="on-push",
        blocking_policy="blocking",
        command="cargo deny --format json --manifest-path engine/Cargo.toml check --config engine/deny.toml licenses",
        inputs=(
            _input(
                path="Cargo.lock",
                purpose="Dependency graph for license/dependency policy checks",
            ),
            _input(path="engine/deny.toml", purpose="Policy configuration"),
        ),
        owner="Rust security maintainer",
        noise="low",
        pinned_version="cargo-deny==0.19.6",
        allowed_to_fail_rationale=None,
    ),
    ScannerManifestEntry(
        name="gitleaks",
        ecosystem="code",
        cadence="on-push",
        blocking_policy="allowed_to_fail",
        command="python tools/security_scan/run_gitleaks_scanners.py --output-dir .",
        inputs=(
            _input(
                path=".gitleaks.toml",
                purpose="Optional allowed secret-pattern baseline",
                required=False,
            ),
            _input(
                path="tools/security_scan/run_gitleaks_scanners.py",
                purpose="Owned Gitleaks release-evidence runner",
            ),
            _input(path=".", purpose="Tracked repository content"),
        ),
        owner="SC-NeuroCore security lane owner",
        noise="medium",
        pinned_version="gitleaks==8.20.1",
        allowed_to_fail_rationale="Gitleaks false positive noise is expected during docs and test fixture churn; triage is performed weekly.",
    ),
    ScannerManifestEntry(
        name="semgrep",
        ecosystem="code",
        cadence="on-push",
        blocking_policy="blocking",
        command="python tools/security_scan/run_semgrep_scanners.py --output-dir .",
        inputs=(
            _input(
                path=".semgrep.yml",
                purpose="Repository-owned Semgrep release policy",
            ),
            _input(
                path="tools/security_scan/run_semgrep_scanners.py",
                purpose="Owned Semgrep release-evidence runner",
            ),
            _input(path="src", purpose="Python source tree"),
            _input(path="tools", purpose="Security and release tooling"),
        ),
        owner="SC-NeuroCore security lane owner",
        noise="low",
        pinned_version="semgrep==1.168.0",
        allowed_to_fail_rationale=None,
    ),
    ScannerManifestEntry(
        name="trivy fs",
        ecosystem="containers",
        cadence="on-push",
        blocking_policy="blocking",
        command="python tools/security_scan/run_trivy_fs_scanners.py --output-dir .",
        inputs=(
            _input(
                path="tools/security_scan/run_trivy_fs_scanners.py",
                purpose="Owned Trivy filesystem release-evidence runner",
            ),
            _input(
                path=".",
                purpose="Repository tree for filesystem vulnerability scan",
            ),
        ),
        owner="SC-NeuroCore security lane owner",
        noise="medium",
        pinned_version="trivy==0.58.1",
        allowed_to_fail_rationale=None,
    ),
    ScannerManifestEntry(
        name="syft-cyclonedx",
        ecosystem="supply-chain",
        cadence="on-push",
        blocking_policy="blocking",
        command=(
            "python tools/security_scan/run_syft_cyclonedx_scanners.py "
            "--output-dir security/ci-security-packet"
        ),
        inputs=(
            _input(path="requirements/release.txt", purpose="Reproducibility baseline"),
            _input(path="Cargo.lock", purpose="Rust dependency baseline", required=False),
        ),
        owner="SC-NeuroCore security lane owner",
        noise="low",
        pinned_version="syft==1.20.0",
        allowed_to_fail_rationale=None,
    ),
    ScannerManifestEntry(
        name="reuse",
        ecosystem="docs",
        cadence="on-push",
        blocking_policy="allowed_to_fail",
        command="reuse --root . lint --json",
        inputs=(
            _input(path="REUSE.toml", purpose="Reuse lint config"),
            _input(path="README.md", purpose="Top-level SPDX-referenced docs"),
            _input(path="tools", purpose="Tooling source coverage"),
        ),
        owner="SC-NeuroCore docs compliance owner",
        noise="low",
        pinned_version="reuse==6.2.0",
        allowed_to_fail_rationale=(
            "REUSE lint currently exposes repo-wide legacy SPDX coverage debt; "
            "CI records the JSON artefact while remediation is tracked separately."
        ),
    ),
    ScannerManifestEntry(
        name="actionlint",
        ecosystem="github",
        cadence="on-push",
        blocking_policy="blocking",
        command="actionlint -format '{{json .}}'",
        inputs=(_input(path=".github/workflows", purpose="Workflow syntax and action policy"),),
        owner="SC-NeuroCore CI lane owner",
        noise="low",
        pinned_version="actionlint==1.7.12",
        allowed_to_fail_rationale=None,
    ),
    ScannerManifestEntry(
        name="pyright",
        ecosystem="python",
        cadence="on-push",
        blocking_policy="blocking",
        command="pyright --project pyrightconfig.json --outputjson",
        inputs=(
            _input(path="pyrightconfig.json", purpose="Type-check policy definition"),
            _input(path="src", purpose="Python static type surface"),
        ),
        owner="SC-NeuroCore typing owner",
        noise="low",
        pinned_version="pyright==1.1.382",
        allowed_to_fail_rationale=None,
    ),
    ScannerManifestEntry(
        name="mypy",
        ecosystem="python",
        cadence="on-push",
        blocking_policy="blocking",
        command="mypy --strict --json-report security/mypy .",
        inputs=(
            _input(path="pyproject.toml", purpose="Mypy config and source discovery"),
            _input(path="src", purpose="Python typed surface"),
        ),
        owner="SC-NeuroCore typing owner",
        noise="low",
        pinned_version="mypy==1.15.0",
        allowed_to_fail_rationale=None,
    ),
    ScannerManifestEntry(
        name="bandit",
        ecosystem="python",
        cadence="on-push",
        blocking_policy="blocking",
        command=(
            "bandit -q -r src/sc_neurocore tools "
            "-x src/sc_neurocore/accel/mojo/.pixi "
            "--severity-level medium --format json --output security/bandit.json"
        ),
        inputs=(_input(path="src", purpose="Python code surface to lint"),),
        owner="SC-NeuroCore code quality owner",
        noise="low",
        pinned_version="bandit==1.9.4",
        allowed_to_fail_rationale=None,
    ),
    ScannerManifestEntry(
        name="ruff",
        ecosystem="python",
        cadence="on-push",
        blocking_policy="blocking",
        command=(
            "ruff check --output-format json --output-file security/ruff.json "
            "--cache-dir security/ruff-cache src tools tests"
        ),
        inputs=(
            _input(path="pyproject.toml", purpose="Ruff rule settings"),
            _input(path="src", purpose="Python source surface"),
        ),
        owner="SC-NeuroCore code quality owner",
        noise="low",
        pinned_version="ruff==0.8.1",
        allowed_to_fail_rationale=None,
    ),
    ScannerManifestEntry(
        name="benchmark-regression",
        ecosystem="python",
        cadence="nightly",
        blocking_policy="blocking",
        command=(
            "python tools/security_scan/run_benchmark_regression_scanners.py "
            "--baseline benchmarks/baselines/security_side_channel_benchmark.json "
            "--current security/benchmark-current/security_side_channel_benchmark.json "
            "--output security/benchmark_regression.json --max-regression-pct 5.0"
        ),
        inputs=(
            _input(
                path="benchmarks/baselines/security_side_channel_benchmark.json",
                purpose="Tracked deterministic side-channel benchmark baseline",
            ),
            _input(
                path="tools/side_channel_benchmark.py",
                purpose="Deterministic current benchmark generator",
            ),
        ),
        owner="SC-NeuroCore performance owner",
        noise="medium",
        pinned_version="internal-check-script-v1",
        allowed_to_fail_rationale=None,
    ),
    ScannerManifestEntry(
        name="cargo-fuzz-nightly",
        ecosystem="rust",
        cadence="nightly",
        blocking_policy="allowed_to_fail",
        command=(
            "python tools/security_scan/run_cargo_fuzz_scanners.py "
            "--output-dir security/ci-security-packet --target all --max-total-time 300"
        ),
        inputs=(
            _input(path="fuzz", purpose="Rust fuzz targets and corpus data"),
            _input(path="src/sc_neurocore", purpose="Rust API and bridge targets"),
        ),
        owner="SC-NeuroCore Rust resilience owner",
        noise="high",
        pinned_version="cargo-fuzz==0.13.2",
        allowed_to_fail_rationale="Nightly fuzzing has known environment-specific flake risk from toolchain and sanitizer noise.",
    ),
)


def build_scanner_manifest() -> dict[str, Any]:
    """Return deterministic scanner manifest payload."""
    scanners = [_manifest_entry_to_dict(entry) for entry in _MANDATORY_SCANNERS]
    return {
        "schema_version": SCAN_MANIFEST_SCHEMA_VERSION,
        "scanners": scanners,
        "metadata": {
            "generated": False,
            "versioned_scanners": len(scanners),
            "scope": "security",
            "source": "tools/security_scanner_manifest.py",
        },
    }


def _manifest_entry_to_dict(entry: ScannerManifestEntry) -> dict[str, Any]:
    return {
        **asdict(entry),
        "inputs": [asdict(scanner_input) for scanner_input in entry.inputs],
    }


def validate_scanner_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate manifest contract and return a findings report."""
    findings: list[dict[str, str]] = []

    if payload.get("schema_version") != SCAN_MANIFEST_SCHEMA_VERSION:
        findings.append(
            {
                "level": "error",
                "message": "schema_version is missing or not equal to expected version",
            }
        )

    scanners = payload.get("scanners")
    if not isinstance(scanners, list):
        findings.append({"level": "error", "message": "scanners must be a list"})
        return _final_report(findings)

    required_fields = {
        "name",
        "ecosystem",
        "cadence",
        "blocking_policy",
        "command",
        "inputs",
        "owner",
        "noise",
        "pinned_version",
    }
    names: set[str] = set()

    for scanner in scanners:
        if not isinstance(scanner, dict):
            findings.append({"level": "error", "message": "scanner entry must be an object"})
            continue
        missing = sorted(required_fields - set(scanner))
        if missing:
            findings.append(
                {
                    "level": "error",
                    "message": f"scanner {scanner.get('name', '<unknown>')} missing fields: {', '.join(missing)}",
                }
            )
        name = scanner.get("name")
        if isinstance(name, str):
            if name in names:
                findings.append(
                    {
                        "level": "error",
                        "message": f"duplicate scanner name {name}",
                    }
                )
            names.add(name)

        policy = scanner.get("blocking_policy")
        rationale = scanner.get("allowed_to_fail_rationale")
        if policy == "allowed_to_fail" and not isinstance(rationale, str):
            findings.append(
                {
                    "level": "error",
                    "message": f"scanner {name} uses allowed_to_fail but has no rationale",
                }
            )

        inputs = scanner.get("inputs", ())
        if not isinstance(inputs, list) or len(inputs) == 0:
            findings.append(
                {
                    "level": "warning",
                    "message": f"scanner {name} has no declared inputs",
                }
            )
        else:
            for scanner_input in inputs:
                if not isinstance(scanner_input, dict):
                    findings.append(
                        {
                            "level": "error",
                            "message": f"scanner {name} input entry must be object",
                        }
                    )
                    continue
                if not isinstance(scanner_input.get("path"), str):
                    findings.append(
                        {
                            "level": "error",
                            "message": f"scanner {name} input.path must be string",
                        }
                    )

    for required in _required_scanner_names():
        if required not in names:
            findings.append(
                {
                    "level": "error",
                    "message": f"required scanner missing from manifest: {required}",
                }
            )

    return _final_report(findings)


def _required_scanner_names() -> set[str]:
    return {
        "pip-audit",
        "osv-scanner",
        "cargo-audit",
        "cargo-deny",
        "gitleaks",
        "semgrep",
        "trivy fs",
        "syft-cyclonedx",
        "reuse",
        "actionlint",
        "pyright",
        "mypy",
        "bandit",
        "ruff",
        "benchmark-regression",
        "cargo-fuzz-nightly",
    }


def _final_report(findings: list[dict[str, str]]) -> dict[str, Any]:
    errors = [finding for finding in findings if finding["level"] == "error"]
    return {
        "passed": len(errors) == 0,
        "findings": findings,
        "errors": len(errors),
        "warnings": len([finding for finding in findings if finding["level"] == "warning"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and validate a deterministic security scanner manifest."
    )
    parser.add_argument("--output", type=Path, help="Write manifest JSON to path")
    parser.add_argument(
        "--validate",
        type=Path,
        help="Validate manifest JSON from path and exit non-zero on failures",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.validate is not None:
        payload = json.loads(args.validate.read_text(encoding="utf-8"))
        report = validate_scanner_manifest(payload)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["passed"] else 1

    manifest = build_scanner_manifest()
    payload = json.dumps(manifest, indent=2, sort_keys=True)

    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
        return 0

    print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
