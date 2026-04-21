# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for experimental report validation

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.validate_experimental_reports import validate_report


REPO_ROOT = Path(
    "/media/anulum/724AA8E84AA8AA75/aaa_God_of_the_Math_Collection/03_CODE/SC-NEUROCORE"
)
VALIDATOR = REPO_ROOT / "tools" / "validate_experimental_reports.py"


def _report_template() -> dict:
    return {
        "route_name": "demo.route",
        "mode": "shadow",
        "total_cases": 2,
        "matched_cases": 2,
        "candidate_failures": 0,
        "median_baseline_runtime_ns": 1000,
        "median_candidate_runtime_ns": 500,
        "cases": [
            {
                "case_name": "a",
                "route_name": "demo.route",
                "returned_path": "shadow-baseline",
                "baseline_runtime_ns": 1100,
                "candidate_runtime_ns": 600,
                "candidate_error": None,
                "comparison": {
                    "matched": True,
                    "comparable_leaf_count": 1,
                    "max_abs_diff": 1e-4,
                    "max_rel_diff": 1e-3,
                    "detail": "ok",
                },
            },
            {
                "case_name": "b",
                "route_name": "demo.route",
                "returned_path": "shadow-baseline",
                "baseline_runtime_ns": 900,
                "candidate_runtime_ns": 400,
                "candidate_error": None,
                "comparison": {
                    "matched": True,
                    "comparable_leaf_count": 1,
                    "max_abs_diff": 2e-4,
                    "max_rel_diff": 2e-3,
                    "detail": "ok",
                },
            },
        ],
    }


def test_validate_report_passes_with_matching_cases():
    report = _report_template()

    result = validate_report(
        report,
        path=Path("benchmarks/results/demo.json"),
        max_abs_diff=1e-3,
        max_rel_diff=1e-2,
        require_mode="shadow",
    )

    assert result.ok
    assert result.reasons == []


def test_validate_report_fails_on_candidate_failures_and_drift():
    report = _report_template()
    report["candidate_failures"] = 1
    report["matched_cases"] = 1
    report["cases"][1]["comparison"]["matched"] = False
    report["cases"][1]["comparison"]["max_abs_diff"] = 0.5
    report["cases"][1]["comparison"]["max_rel_diff"] = 0.25
    report["cases"][1]["candidate_error"] = "boom"

    result = validate_report(
        report,
        path=Path("benchmarks/results/demo.json"),
        max_abs_diff=0.01,
        max_rel_diff=0.01,
        require_mode="shadow",
    )

    assert not result.ok
    assert any("candidate_failures=1" in reason for reason in result.reasons)
    assert any("matched_cases=1/2" in reason for reason in result.reasons)
    assert any("max_abs_diff=0.5 > 0.01" in reason for reason in result.reasons)
    assert any("max_rel_diff=0.25 > 0.01" in reason for reason in result.reasons)


def test_validator_cli_passes_on_real_reports():
    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--require-mode",
            "shadow",
            "--max-abs-diff",
            "0.01",
            "--max-rel-diff",
            "0.01",
            "benchmarks/results/experimental_physics_heat_cosine_mode.json",
            "benchmarks/results/experimental_physics_oscillator_harmonic_symplectic.json",
            "benchmarks/results/experimental_solver_lif_subthreshold_exact.json",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=20,
    )

    assert result.returncode == 0
    assert "PASS physics.heat.cosine-mode" in result.stdout


def test_validator_cli_json_output(tmp_path):
    report_path = tmp_path / "experimental_demo.json"
    report_path.write_text(json.dumps(_report_template()))

    result = subprocess.run(
        [sys.executable, str(VALIDATOR), "--json", str(report_path)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=20,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload[0]["route_name"] == "demo.route"
    assert payload[0]["ok"] is True
