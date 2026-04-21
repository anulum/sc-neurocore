# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for experimental suite comparison

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.compare_experimental_suites import compare_suites


REPO_ROOT = Path(
    "/media/anulum/724AA8E84AA8AA75/aaa_God_of_the_Math_Collection/03_CODE/SC-NEUROCORE"
)
COMPARE = REPO_ROOT / "tools" / "compare_experimental_suites.py"


def _write_suite(path: Path, routes: list[dict]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "suite_summary.json").write_text(
        json.dumps(
            {
                "output_dir": str(path),
                "routes": routes,
            }
        )
    )


def test_compare_suites_reports_common_and_unique_routes(tmp_path):
    base = tmp_path / "base"
    cand = tmp_path / "cand"
    _write_suite(
        base,
        [
            {
                "route_name": "physics.a",
                "runs": 2,
                "all_passed": True,
                "max_abs_diff": 0.1,
                "max_rel_diff": 0.2,
            },
            {
                "route_name": "physics.only_base",
                "runs": 2,
                "all_passed": True,
                "max_abs_diff": 0.0,
                "max_rel_diff": 0.0,
            },
        ],
    )
    _write_suite(
        cand,
        [
            {
                "route_name": "physics.a",
                "runs": 3,
                "all_passed": True,
                "max_abs_diff": 0.15,
                "max_rel_diff": 0.25,
            },
            {
                "route_name": "physics.only_cand",
                "runs": 3,
                "all_passed": True,
                "max_abs_diff": 0.0,
                "max_rel_diff": 0.0,
            },
        ],
    )

    result = compare_suites(base, cand)

    assert result["common_routes"][0]["route_name"] == "physics.a"
    assert result["common_routes"][0]["delta_max_abs_diff"] == pytest.approx(0.05)
    assert result["baseline_only_routes"] == ["physics.only_base"]
    assert result["candidate_only_routes"] == ["physics.only_cand"]


def test_compare_suites_cli_json(tmp_path):
    base = tmp_path / "base"
    cand = tmp_path / "cand"
    _write_suite(
        base,
        [
            {
                "route_name": "physics.a",
                "runs": 1,
                "all_passed": True,
                "max_abs_diff": 0.1,
                "max_rel_diff": 0.2,
            }
        ],
    )
    _write_suite(
        cand,
        [
            {
                "route_name": "physics.a",
                "runs": 1,
                "all_passed": True,
                "max_abs_diff": 0.11,
                "max_rel_diff": 0.21,
            }
        ],
    )

    result = subprocess.run(
        [sys.executable, str(COMPARE), "--json", str(base), str(cand)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=20,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["common_routes"][0]["route_name"] == "physics.a"
