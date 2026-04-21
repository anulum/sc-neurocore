# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for experimental suite execution

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.run_experimental_suite import run_suite


REPO_ROOT = Path(
    "/media/anulum/724AA8E84AA8AA75/aaa_God_of_the_Math_Collection/03_CODE/SC-NEUROCORE"
)
SUITE_RUNNER = REPO_ROOT / "tools" / "run_experimental_suite.py"


def test_run_suite_writes_summary_and_reports(tmp_path):
    summary = run_suite(
        repetitions=1,
        mode="shadow",
        max_abs_diff=0.01,
        max_rel_diff=0.01,
        output_dir=tmp_path,
    )

    assert summary["all_passed"] is True
    assert summary["route_count"] >= 3
    assert (tmp_path / "suite_summary.json").exists()
    assert (tmp_path / "suite_summary.md").exists()
    reports = sorted(tmp_path.glob("rep_01_experimental_*.json"))
    assert len(reports) >= 3


def test_run_suite_real_only_excludes_demo_route(tmp_path):
    summary = run_suite(
        repetitions=1,
        mode="shadow",
        max_abs_diff=0.01,
        max_rel_diff=0.01,
        output_dir=tmp_path,
        real_only=True,
    )

    assert summary["all_passed"] is True
    assert summary["real_only"] is True
    assert all(not route["route_name"].startswith("demo.") for route in summary["routes"])
    reports = sorted(tmp_path.glob("rep_01_experimental_*.json"))
    assert all("demo_affine_sigmoid" not in report.name for report in reports)


def test_suite_runner_cli(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SUITE_RUNNER),
            "--repetitions",
            "1",
            "--mode",
            "shadow",
            "--max-abs-diff",
            "0.01",
            "--max-rel-diff",
            "0.01",
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={"PYTHONPATH": "src"},
        timeout=60,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["all_passed"] is True
    assert payload["output_dir"] == str(tmp_path)


def test_suite_runner_cli_real_only(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SUITE_RUNNER),
            "--repetitions",
            "1",
            "--mode",
            "shadow",
            "--max-abs-diff",
            "0.01",
            "--max-rel-diff",
            "0.01",
            "--real-only",
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={"PYTHONPATH": "src"},
        timeout=60,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["all_passed"] is True
    assert payload["real_only"] is True
    assert all(not route["route_name"].startswith("demo.") for route in payload["routes"])
