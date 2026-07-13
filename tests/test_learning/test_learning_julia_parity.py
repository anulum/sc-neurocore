# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia autonomous-learning parity tests

"""Source contracts and live parity for the Julia autonomous-learning bridge."""

from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess

import pytest

from sc_neurocore._native.learning_bridge import RULE_STDP, RustPlasticityRule, is_available

REPO_ROOT = Path(__file__).resolve().parents[2]
JULIA_BRIDGE = REPO_ROOT / "src/sc_neurocore/accel/julia/_native/learning_bridge.jl"
FFI_AVAILABLE = is_available()
JULIA_SCRIPT = """
include("src/sc_neurocore/accel/julia/_native/learning_bridge.jl")
using .LearningBridgeAccel

if !LearningBridgeAccel.is_available()
    println("NO_FFI")
    exit()
end

rule = LearningBridgeAccel.RustPlasticityRule(
    LearningBridgeAccel.RULE_STDP, 0.5f0, 0.1f0, 0.05f0
)
LearningBridgeAccel.step(rule, true, false, 0.0f0)
LearningBridgeAccel.step(rule, false, true, 0.0f0)

try
    LearningBridgeAccel.step_batched(rule, Bool[true], Bool[false, true], Float32[0.0])
    println("UNSAFE_BATCH_ACCEPTED")
catch error
    println(error isa DimensionMismatch ? "SAFE_BATCH_REJECTED" : "WRONG_BATCH_ERROR")
end

println(LearningBridgeAccel.weight(rule))
LearningBridgeAccel.destroy_rule(rule)
"""


def _run_julia() -> subprocess.CompletedProcess[str]:
    """Execute Julia from the repository root with the current native path."""
    try:
        return subprocess.run(
            ["julia", "--startup-file=no", "-e", JULIA_SCRIPT],
            cwd=REPO_ROOT,
            env=dict(os.environ),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        pytest.skip("Julia toolchain is not installed")


def _julia_weight(result: subprocess.CompletedProcess[str]) -> float:
    """Parse a successful, length-safe Julia parity result."""
    assert result.returncode == 0, (
        "Julia autonomous-learning script failed.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    lines = result.stdout.strip().splitlines()
    assert lines and lines[-1] != "NO_FFI", "Julia could not load the Rust FFI"
    assert "SAFE_BATCH_REJECTED" in lines
    assert "UNSAFE_BATCH_ACCEPTED" not in lines
    try:
        return float(lines[-1])
    except ValueError as exc:
        raise AssertionError(f"Julia output is not a weight: {lines[-1]!r}") from exc


def test_julia_bridge_uses_explicit_library_override() -> None:
    """Lock Julia parity to the same fresh library as Python and Go."""
    source = JULIA_BRIDGE.read_text(encoding="utf-8")
    assert 'get(ENV, "SC_NEUROCORE_LIB_PATH", _DEFAULT_LIB_PATH)' in source


def test_julia_bridge_validates_constructor_and_step_domains() -> None:
    """Lock rule, finite-number, timestep, and live-handle checks."""
    source = JULIA_BRIDGE.read_text(encoding="utf-8")
    assert "require_rule_type(rule_type)" in source
    assert "require_weight(weight)" in source
    assert 'require_positive("dt", dt)' in source
    assert 'require_live("plasticity rule", s.ptr)' in source
    assert "ptr != C_NULL || error" in source


def test_julia_bridge_validates_equal_nonempty_batches() -> None:
    """Prevent Julia vectors from exposing out-of-bounds Rust slices."""
    source = JULIA_BRIDGE.read_text(encoding="utf-8")
    assert "count = require_equal_nonempty(pre_spikes, post_spikes, rewards)" in source
    assert "count = require_equal_nonempty(fired, pre_spikes, global_rewards)" in source
    assert "learning vectors must have equal lengths" in source


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (subprocess.CompletedProcess(["julia"], 1, "", "compile failed"), "failed"),
        (subprocess.CompletedProcess(["julia"], 0, "", ""), "could not load"),
        (subprocess.CompletedProcess(["julia"], 0, "NO_FFI\n", ""), "could not load"),
        (
            subprocess.CompletedProcess(["julia"], 0, "UNSAFE_BATCH_ACCEPTED\n0.5\n", ""),
            "SAFE_BATCH_REJECTED",
        ),
        (
            subprocess.CompletedProcess(["julia"], 0, "SAFE_BATCH_REJECTED\nnot-float\n", ""),
            "not a weight",
        ),
    ],
)
def test_julia_weight_rejects_broken_results(
    result: subprocess.CompletedProcess[str], message: str
) -> None:
    """Keep setup and safety failures visible instead of silently skipping."""
    with pytest.raises(AssertionError, match=message):
        _julia_weight(result)


def test_run_julia_reports_missing_toolchain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retain the explicit optional-toolchain boundary."""

    def missing(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("julia")

    monkeypatch.setattr(subprocess, "run", missing)
    with pytest.raises(pytest.skip.Exception, match="not installed"):
        _run_julia()


@pytest.mark.skipif(not FFI_AVAILABLE, reason="Rust FFI not available")
def test_julia_python_learning_parity() -> None:
    """Compare Julia and Python over the same two-event STDP trace."""
    rule = RustPlasticityRule(rule_type=RULE_STDP, weight=0.5, param_a=0.1, param_b=0.05)
    rule.step(True, False)
    rule.step(False, True)
    julia_weight = _julia_weight(_run_julia())
    assert math.isclose(rule.weight, julia_weight, rel_tol=1e-5)
