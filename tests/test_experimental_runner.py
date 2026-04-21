# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Smoke tests for the experimental alternative-path runner

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


# Derive the repo root from this file's location — a hard-coded absolute
# path would work on the author's workstation only; CI runners, wheels
# installed into virtualenvs, and anyone else cloning the repo elsewhere
# all need the relative walk.
REPO_ROOT = str(Path(__file__).resolve().parent.parent)
RUNNER = f"{REPO_ROOT}/tools/run_experimental_path.py"


def _runner_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT}/src" if not existing else f"{REPO_ROOT}/src:{existing}"
    return env


def test_runner_lists_routes():
    result = subprocess.run(
        [sys.executable, RUNNER, "--list-routes"],
        capture_output=True,
        text=True,
        timeout=20,
        cwd=REPO_ROOT,
        env=_runner_env(),
    )

    assert result.returncode == 0
    assert "demo.affine-sigmoid" in result.stdout
    assert "physics.heat.cosine-mode" in result.stdout
    assert "physics.oscillator.harmonic-symplectic" in result.stdout
    assert "physics.kuramoto.noiseless-symplectic-lift" in result.stdout
    assert "solver.lif.subthreshold-exact" in result.stdout


def test_runner_writes_demo_report(tmp_path):
    out = tmp_path / "experimental_demo.json"
    result = subprocess.run(
        [
            sys.executable,
            RUNNER,
            "--route",
            "demo.affine-sigmoid",
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        cwd=REPO_ROOT,
        env=_runner_env(),
    )

    assert result.returncode == 0
    assert out.exists()
    text = out.read_text()
    assert '"route_name": "demo.affine-sigmoid"' in text
