"""Timing-aware formal property framework workflow contract tests.

This file verifies the NEU-C.2 workflow contract: a bounded timing property is
represented once, emitted deterministically for external model-checker surfaces,
and connected to a concrete dense-layer formal proof without mocking the RTL
unit under test.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "hdl" / "formal"))

from timing import (  # noqa: E402
    TimingProofOrchestrator,
    TimingProperty,
    emit_kind2_module,
    emit_nuxmv_module,
)

EXAMPLE_SBY = REPO_ROOT / "hdl" / "formal" / "timing" / "example_dense_layer_core_latency.sby"


def test_emitters_encode_bounded_timing_contract() -> None:
    prop = TimingProperty(
        name="dense_start_to_done",
        kind="latency",
        trigger="start_pulse",
        response="run_done",
        bound_cycles=6,
    )

    nuxmv_model = emit_nuxmv_module(prop)
    kind2_model = emit_kind2_module(prop)

    assert "MODULE main" in nuxmv_model
    assert "age : 0..6" in nuxmv_model
    assert "INVARSPEC !violation" in nuxmv_model
    assert "node dense_start_to_done" in kind2_model
    assert "--%PROPERTY ok;" in kind2_model
    assert "pre_age >= 6" in kind2_model


def test_timing_property_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="bound_cycles"):
        TimingProperty(
            name="dense_bad_bound",
            kind="deadline",
            trigger="start_pulse",
            response="run_done",
            bound_cycles=-1,
        )


def test_orchestrator_reports_missing_external_dependency() -> None:
    orchestrator = TimingProofOrchestrator(
        EXAMPLE_SBY,
        executable="sc_neurocore_missing_sby",
        solver="sc_neurocore_missing_solver",
    )

    result = orchestrator.prove()

    assert result.passed is False
    assert result.exit_code == 127
    assert result.unavailable == ("sc_neurocore_missing_sby", "sc_neurocore_missing_solver")
    assert "missing formal dependency" in result.stderr_tail


@pytest.mark.skipif(
    shutil.which("sby") is None or shutil.which("cvc5") is None,
    reason="SymbiYosys/cvc5 external formal dependencies are unavailable",
)
def test_dense_layer_latency_example_proves_with_symbiyosys(tmp_path: Path) -> None:
    result = TimingProofOrchestrator(EXAMPLE_SBY, temp_root=tmp_path).prove(timeout_s=120)

    assert result.passed, result.stdout_tail + "\n" + result.stderr_tail
    assert result.exit_code == 0
    assert result.unavailable == ()
