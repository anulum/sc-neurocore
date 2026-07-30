# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.

"""Custody tests for the complete documented Go SC Compte network lane."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

REPOSITORY = Path(__file__).resolve().parents[1]
GO_ROOT = REPOSITORY / "src/sc_neurocore/accel/go"
SOURCE = GO_ROOT / "sc_compte_wm_network/network.go"
NATIVE_TEST = GO_ROOT / "sc_compte_wm_network/network_test.go"
BENCHMARK = GO_ROOT / "cmd/bench_sc_compte_wm_network/main.go"
GO_MOD = GO_ROOT / "go.mod"
RESULT = REPOSITORY / "benchmarks/results/bench_sc_compte_wm_network_go.json"
PYTHON_RESULT = REPOSITORY / "benchmarks/results/bench_sc_compte_wm_network.json"


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    go_home = REPOSITORY / ".venv/go"
    environment.update(
        {
            "GOPATH": str(go_home),
            "GOMODCACHE": str(go_home / "pkg/mod"),
            "GOCACHE": str(go_home / "cache"),
            "GOTOOLCHAIN": "auto",
        }
    )
    return environment


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_go_network_native_parity_suite() -> None:
    executed = subprocess.run(
        ["go", "test", "./sc_compte_wm_network", "-count=1"],
        cwd=GO_ROOT,
        env=_environment(),
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert executed.returncode == 0, executed.stdout + executed.stderr
    assert "github.com/anulum/sc-neurocore/accel/sc_compte_wm_network" in executed.stdout


def test_go_network_public_surface_has_godoc() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    for declaration in (
        "type Spec struct",
        "type State struct",
        "type StepReceipt struct",
        "type Network struct",
        "func NewNetwork(",
        "func ValidateState(",
        "func CounterPoissonCounts(",
        "func (network *Network) StepWithEvents(",
        "func (network *Network) Run(",
    ):
        offset = source.index(declaration)
        assert "// " in source[max(0, offset - 500) : offset]
    documented = subprocess.run(
        ["go", "doc", "./sc_compte_wm_network"],
        cwd=GO_ROOT,
        env=_environment(),
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert documented.returncode == 0, documented.stderr
    assert "SC-COMPTE-WM-NETWORK" in documented.stdout


def test_go_benchmark_receipt_is_source_bound_and_matches_python() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    python = json.loads(PYTHON_RESULT.read_text(encoding="utf-8"))
    assert payload["model"] == "SC-COMPTE-WM-NETWORK"
    assert payload["execution_path"] == "go-midpoint-rk2-radix2-fft"
    assert payload["configuration"]["cells"] == 2560
    assert payload["configuration"]["steps"] == 1000
    assert payload["configuration"]["repeats"] == 3
    assert payload["repeat_receipts_exact"] is True
    assert payload["passed"] is True
    assert payload["persistent_bump_claimed"] is False
    assert payload["distractor_resistance_claimed"] is False
    assert payload["input_sha256"] == python["input_sha256"]
    assert payload["spike_sha256"] == python["spike_sha256"]
    assert payload["spike_counts"] == python["spike_counts"]
    for path in (GO_MOD, SOURCE, BENCHMARK):
        relative = path.relative_to(REPOSITORY).as_posix()
        assert payload["source_sha256"][relative] == _sha256(path)
