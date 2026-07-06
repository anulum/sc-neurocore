# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Security Scanner Workflow CI Wiring

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import yaml


def _load_workflow() -> dict[str, Any]:
    workflow_path = (
        Path(__file__).resolve().parents[2] / ".github" / "workflows" / "security-scanners.yml"
    )
    assert workflow_path.exists()
    return cast(dict[str, Any], yaml.safe_load(workflow_path.read_text(encoding="utf-8")))


def _workflow_events(workflow: dict[str, Any]) -> dict[str, Any]:
    raw_events = workflow.get("on")
    if raw_events is None:
        raw_events = cast(dict[Any, Any], workflow).get(True, {})
    return cast(dict[str, Any], raw_events)


def test_security_scanner_workflow_triggers_and_scope() -> None:
    workflow = _load_workflow()
    on = _workflow_events(workflow)

    assert "workflow_dispatch" in on
    assert "pull_request" in on
    assert "push" in on
    assert "schedule" in on

    push_paths = set(on["push"].get("paths", []))
    pull_request_paths = set(on["pull_request"].get("paths", []))
    for required in {
        ".github/workflows/security-scanners.yml",
        "security/**",
        "tools/security_scan/**",
        "requirements/**",
        "tests/test_tools/test_run_gitleaks_scanners.py",
        "tests/test_tools/test_run_trivy_fs_scanners.py",
    }:
        assert required in push_paths
        assert required in pull_request_paths


def test_security_scanner_workflow_invokes_packet_builder() -> None:
    workflow = _load_workflow()
    jobs = workflow["jobs"]
    assert jobs

    scan_job = jobs["security-scanner-manifest"]
    steps = scan_job["steps"]

    run_steps = [step["run"] for step in steps if isinstance(step, dict) and "run" in step]
    assert any("tools/security_scan/ci_security_packet.py" in step for step in run_steps)
    assert any(
        "--output-dir" in step and "security/ci-security-packet" in step for step in run_steps
    )
    assert any("--fail-on-missing-required" in step for step in run_steps)
    packet_steps = [
        step for step in run_steps if "tools/security_scan/ci_security_packet.py" in step
    ]
    assert not any("--output security/ci-security-packet" in step for step in packet_steps)


def test_security_scanner_workflow_runs_lightweight_scanner_lane() -> None:
    workflow = _load_workflow()
    steps = workflow["jobs"]["security-scanner-manifest"]["steps"]

    run_text = "\n".join(step["run"] for step in steps if isinstance(step, dict) and "run" in step)

    assert "tools/security_scan/run_lightweight_security_scanners.py" in run_text
    assert "--output-dir security/ci-security-packet" in run_text
    assert "security/ci-security-packet/security/lightweight_scanner_summary.json" in run_text
    assert "tools/security_scan/release_security_artifact_index.py" in run_text
    assert "--root security/ci-security-packet" in run_text


def test_security_scanner_workflow_runs_rust_scanner_lane() -> None:
    workflow = _load_workflow()
    steps = workflow["jobs"]["security-scanner-manifest"]["steps"]

    run_text = "\n".join(step["run"] for step in steps if isinstance(step, dict) and "run" in step)

    assert "cargo install cargo-audit --version 0.22.1 --locked" in run_text
    assert "cargo install cargo-deny --version 0.19.6 --locked" in run_text
    assert "tools/security_scan/run_rust_security_scanners.py" in run_text
    assert "security/ci-security-packet/security/rust_scanner_summary.json" in run_text


def test_security_scanner_workflow_runs_python_compliance_lane() -> None:
    workflow = _load_workflow()
    steps = workflow["jobs"]["security-scanner-manifest"]["steps"]

    run_text = "\n".join(step["run"] for step in steps if isinstance(step, dict) and "run" in step)

    assert (
        "python -m pip install --require-hashes -r requirements/security-scanners.txt" in run_text
    )
    assert "tools/security_scan/run_python_compliance_scanners.py" in run_text
    assert "security/ci-security-packet/security/python_compliance_summary.json" in run_text


def test_security_scanner_workflow_runs_osv_scanner_lane() -> None:
    workflow = _load_workflow()
    steps = workflow["jobs"]["security-scanner-manifest"]["steps"]

    run_text = "\n".join(step["run"] for step in steps if isinstance(step, dict) and "run" in step)
    setup_go_versions = [
        step.get("with", {}).get("go-version")
        for step in steps
        if isinstance(step, dict) and str(step.get("uses", "")).startswith("actions/setup-go@")
    ]

    assert "1.26.3" in setup_go_versions
    assert "github.com/google/osv-scanner/v2/cmd/osv-scanner@v2.3.8" in run_text
    assert "tools/security_scan/run_osv_scanners.py" in run_text
    assert "security/ci-security-packet/security/osv_scanner.json" in run_text
    assert "security/ci-security-packet/security/osv_scanner_summary.json" in run_text


def test_security_scanner_workflow_runs_syft_cyclonedx_lane() -> None:
    workflow = _load_workflow()
    steps = workflow["jobs"]["security-scanner-manifest"]["steps"]

    run_text = "\n".join(step["run"] for step in steps if isinstance(step, dict) and "run" in step)

    assert "go install github.com/anchore/syft/cmd/syft@v1.20.0" in run_text
    assert "tools/security_scan/run_syft_cyclonedx_scanners.py" in run_text
    assert "security/ci-security-packet/security/sbom.cdx.json" in run_text
    assert "security/ci-security-packet/security/syft_cyclonedx_summary.json" in run_text


def test_security_scanner_workflow_runs_source_policy_scanner_lanes() -> None:
    workflow = _load_workflow()
    steps = workflow["jobs"]["security-scanner-manifest"]["steps"]

    run_text = "\n".join(step["run"] for step in steps if isinstance(step, dict) and "run" in step)

    assert "python -m pip install --require-hashes -r requirements/semgrep.txt" in run_text
    assert "go install github.com/zricethezav/gitleaks/v8@v8.20.1" in run_text
    assert "go install github.com/aquasecurity/trivy/cmd/trivy@v0.58.1" in run_text

    assert "tools/security_scan/run_semgrep_scanners.py" in run_text
    assert "security/ci-security-packet/security/semgrep.json" in run_text
    assert "security/ci-security-packet/security/semgrep_summary.json" in run_text

    assert "tools/security_scan/run_gitleaks_scanners.py" in run_text
    assert "security/ci-security-packet/security/gitleaks.json" in run_text
    assert "security/ci-security-packet/security/gitleaks_summary.json" in run_text

    assert "tools/security_scan/run_trivy_fs_scanners.py" in run_text
    assert "security/ci-security-packet/security/trivy_fs.json" in run_text
    assert "security/ci-security-packet/security/trivy_fs_summary.json" in run_text


def test_security_scanner_workflow_avoids_unpinned_pip_and_curl_installers() -> None:
    workflow = _load_workflow()
    steps = workflow["jobs"]["security-scanner-manifest"]["steps"]

    run_text = "\n".join(step["run"] for step in steps if isinstance(step, dict) and "run" in step)

    assert "pip install --upgrade pip" not in run_text
    assert "python -m pip install bandit" not in run_text
    assert "curl -sSfL" not in run_text
    assert "install.sh" not in run_text


def test_security_scanner_workflow_runs_cargo_fuzz_only_on_nightly() -> None:
    workflow = _load_workflow()
    jobs = workflow["jobs"]
    fuzz_job = jobs["nightly-cargo-fuzz"]

    assert fuzz_job["if"] == "github.event_name == 'schedule'"

    run_text = "\n".join(
        step["run"] for step in fuzz_job["steps"] if isinstance(step, dict) and "run" in step
    )
    toolchains = [
        step.get("with", {}).get("toolchain")
        for step in fuzz_job["steps"]
        if isinstance(step, dict)
        and str(step.get("uses", "")).startswith("dtolnay/rust-toolchain@")
    ]

    assert "nightly" in toolchains
    assert "cargo install cargo-fuzz --version 0.13.2" in run_text
    assert "cargo-fuzz --version 0.13.1 --locked" not in run_text
    assert "tools/security_scan/run_cargo_fuzz_scanners.py" in run_text
    assert "--target all" in run_text
    assert "--max-total-time 300" in run_text
    assert "security/cargo-fuzz-packet/security/cargo_fuzz_summary.json" in run_text


def test_security_scanner_workflow_runs_benchmark_regression_only_on_nightly() -> None:
    workflow = _load_workflow()
    jobs = workflow["jobs"]
    benchmark_job = jobs["nightly-benchmark-regression"]

    assert benchmark_job["if"] == "github.event_name == 'schedule'"

    run_text = "\n".join(
        step["run"] for step in benchmark_job["steps"] if isinstance(step, dict) and "run" in step
    )

    assert "tools/side_channel_benchmark.py" in run_text
    assert "--output security/benchmark-current/security_side_channel_benchmark.json" in run_text
    assert "tools/security_scan/run_benchmark_regression_scanners.py" in run_text
    assert "--baseline benchmarks/baselines/security_side_channel_benchmark.json" in run_text
    assert "--current security/benchmark-current/security_side_channel_benchmark.json" in run_text
    assert (
        "--output security/benchmark-regression-packet/security/benchmark_regression.json"
        in run_text
    )
    assert "security/benchmark-regression-packet/security/benchmark_regression.json" in run_text


def test_security_scanner_workflow_checks_manifest_consistency_if_present() -> None:
    workflow = _load_workflow()
    steps = workflow["jobs"]["security-scanner-manifest"]["steps"]

    all_runs = [step["run"] for step in steps if isinstance(step, dict) and "run" in step]

    run_text = "\n".join(all_runs)
    for banned in {
        "pyright",
        "mypy",
    }:
        assert banned not in run_text

    manifest_validation = [
        step for step in all_runs if "security_scanner_manifest.py --validate" in step
    ]
    if manifest_validation:
        for command in manifest_validation:
            assert re.search(r"--validate\s+([^\s#]+)", command) is not None


def test_security_scanner_workflow_uploads_artifact() -> None:
    workflow = _load_workflow()
    steps = workflow["jobs"]["security-scanner-manifest"]["steps"]

    assert any(
        isinstance(step, dict)
        and step.get("uses", "").startswith("actions/upload-artifact@")
        and "security/ci-security-packet" in step.get("with", {}).get("path", "")
        for step in steps
    )
