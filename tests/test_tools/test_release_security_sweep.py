# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "security_scan" / "release_security_sweep.py"


def _load_tool() -> Any:
    spec = importlib.util.spec_from_file_location("release_security_sweep", TOOL_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _summary(name: str, *, passed: bool = True) -> dict[str, Any]:
    return {
        "schema_version": f"sc-neurocore.{name}.v1",
        "passed": passed,
        "failed_scanners": [] if passed else [name],
    }


def test_release_security_sweep_runs_real_release_lanes_and_writes_summary(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    calls: list[str] = []

    def record(name: str, *, passed: bool = True) -> dict[str, Any]:
        calls.append(name)
        return _summary(name, passed=passed)

    summary = tool.run_release_security_sweep(
        repo_root=REPO_ROOT,
        output_dir=tmp_path,
        include_fuzz=True,
        fuzz_max_total_time=17,
        build_ci_packet=lambda **_kwargs: record("ci-security-packet"),
        run_lightweight=lambda **_kwargs: record("lightweight-security-scanners"),
        run_python_compliance=lambda **_kwargs: record("python-compliance-scanners"),
        run_osv=lambda **_kwargs: record("osv-scanner"),
        run_rust=lambda **_kwargs: record("rust-security-scanners"),
        run_syft=lambda **_kwargs: record("syft-cyclonedx-scanner"),
        run_semgrep=lambda **_kwargs: record("semgrep-scanner"),
        run_gitleaks=lambda **_kwargs: record("gitleaks-scanner"),
        run_trivy_fs=lambda **_kwargs: record("trivy-fs-scanner"),
        run_supply_chain_audit=lambda **_kwargs: record("supply-chain-audit"),
        run_hypothesis_fuzz=lambda **_kwargs: record("hypothesis-fuzz-subset"),
        run_cargo_fuzz=lambda **kwargs: record(f"cargo-fuzz-{kwargs['max_total_time']}"),
        run_rust_proptest=lambda **_kwargs: record("rust-proptest-subset"),
        build_artifact_index=lambda **_kwargs: record("release-artifact-index"),
    )

    assert calls == [
        "ci-security-packet",
        "lightweight-security-scanners",
        "python-compliance-scanners",
        "osv-scanner",
        "rust-security-scanners",
        "syft-cyclonedx-scanner",
        "semgrep-scanner",
        "gitleaks-scanner",
        "trivy-fs-scanner",
        "supply-chain-audit",
        "hypothesis-fuzz-subset",
        "rust-proptest-subset",
        "cargo-fuzz-17",
        "release-artifact-index",
    ]
    assert summary["schema_version"] == tool.RELEASE_SECURITY_SWEEP_SCHEMA_VERSION
    assert summary["passed"] is True
    assert summary["output_dir"] == str(tmp_path.resolve())
    assert [lane["name"] for lane in summary["lanes"]] == [
        "ci-security-packet",
        "lightweight-security-scanners",
        "python-compliance-scanners",
        "osv-scanner",
        "rust-security-scanners",
        "syft-cyclonedx-scanner",
        "semgrep-scanner",
        "gitleaks-scanner",
        "trivy-fs-scanner",
        "supply-chain-audit",
        "hypothesis-fuzz-subset",
        "rust-proptest-subset",
        "cargo-fuzz",
        "release-artifact-index",
    ]
    assert summary["failed_lanes"] == []

    written = json.loads(
        (tmp_path / "security" / "release_security_sweep_summary.json").read_text(encoding="utf-8")
    )
    assert written == summary


def test_release_security_sweep_fails_when_required_lane_fails(tmp_path: Path) -> None:
    tool = _load_tool()
    summary = tool.run_release_security_sweep(
        repo_root=REPO_ROOT,
        output_dir=tmp_path,
        include_fuzz=False,
        build_ci_packet=lambda **_kwargs: _summary("ci-security-packet"),
        run_lightweight=lambda **_kwargs: _summary("lightweight-security-scanners"),
        run_python_compliance=lambda **_kwargs: _summary(
            "python-compliance-scanners", passed=False
        ),
        run_osv=lambda **_kwargs: _summary("osv-scanner"),
        run_rust=lambda **_kwargs: _summary("rust-security-scanners"),
        run_syft=lambda **_kwargs: _summary("syft-cyclonedx-scanner"),
        run_semgrep=lambda **_kwargs: _summary("semgrep-scanner"),
        run_gitleaks=lambda **_kwargs: _summary("gitleaks-scanner"),
        run_trivy_fs=lambda **_kwargs: _summary("trivy-fs-scanner"),
        run_supply_chain_audit=lambda **_kwargs: _summary("supply-chain-audit"),
        run_hypothesis_fuzz=lambda **_kwargs: _summary("hypothesis-fuzz-subset"),
        run_cargo_fuzz=lambda **_kwargs: _summary("cargo-fuzz"),
        run_rust_proptest=lambda **_kwargs: _summary("rust-proptest-subset"),
        build_artifact_index=lambda **_kwargs: _summary("release-artifact-index"),
    )

    assert summary["passed"] is False
    assert summary["failed_lanes"] == ["python-compliance-scanners"]
    assert [lane["name"] for lane in summary["lanes"]] == [
        "ci-security-packet",
        "lightweight-security-scanners",
        "python-compliance-scanners",
        "osv-scanner",
        "rust-security-scanners",
        "syft-cyclonedx-scanner",
        "semgrep-scanner",
        "gitleaks-scanner",
        "trivy-fs-scanner",
        "supply-chain-audit",
        "hypothesis-fuzz-subset",
        "rust-proptest-subset",
        "release-artifact-index",
    ]


def test_release_security_sweep_cli_returns_nonzero_for_failed_lane(
    tmp_path: Path,
    capsys: Any,
) -> None:
    tool = _load_tool()

    def fake_runner(**_kwargs: Any) -> dict[str, Any]:
        return {
            "schema_version": tool.RELEASE_SECURITY_SWEEP_SCHEMA_VERSION,
            "passed": False,
            "failed_lanes": ["osv-scanner"],
        }

    exit_code = tool.main(
        ["--repo-root", str(REPO_ROOT), "--output-dir", str(tmp_path)],
        runner=fake_runner,
    )

    assert exit_code == 1
    output = json.loads(capsys.readouterr().out)
    assert output["failed_lanes"] == ["osv-scanner"]
