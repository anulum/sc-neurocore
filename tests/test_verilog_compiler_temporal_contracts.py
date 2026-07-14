# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Registered RTL temporal contract tests

"""Focused contracts for edge history and macro-substep RTL state."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from sc_neurocore.compiler.verilog_compiler import compile_to_verilog
from sc_neurocore.neurons.equation_builder import EquationNeuron

HAS_IVERILOG = shutil.which("iverilog") is not None


def _crossing_neuron(*, initial_v: float = 0.0, substeps: int = 1) -> EquationNeuron:
    """Build a non-resetting crossing neuron eligible for registered edge logic."""
    return EquationNeuron(
        equations={"v": "I"},
        parameters={"v_threshold": 1.0},
        state={"v": initial_v},
        threshold="v >= v_threshold",
        detection="crossing",
        dt=1.0,
        method="euler",
        substeps=substeps,
    )


@pytest.mark.parametrize(
    ("initial_v", "expected_history"),
    [(0.0, "1'b0"), (2.0, "1'b1")],
    ids=("initially-inactive", "initially-active"),
)
def test_single_step_crossing_tracks_initial_and_committed_threshold(
    initial_v: float,
    expected_history: str,
) -> None:
    """Edge history starts from the golden state and refreshes after each step."""
    verilog = compile_to_verilog(
        _crossing_neuron(initial_v=initial_v),
        module_name="sc_crossing_history",
    )

    assert "reg _thr_prev;" in verilog
    assert f"_thr_prev <= {expected_history};" in verilog
    assert "&& !_thr_prev" in verilog
    assert "_thr_prev <= (" in verilog
    assert "_ss_cnt" not in verilog


def test_macro_substeps_advance_state_and_sample_crossing_only_at_boundary() -> None:
    """A two-substep macro-step owns a counter and gates its edge decision."""
    verilog = compile_to_verilog(
        _crossing_neuron(substeps=2),
        module_name="sc_crossing_substeps",
    )

    assert "reg [0:0] _ss_cnt;" in verilog
    assert "wire _macro_boundary = (_ss_cnt == 1'd1);" in verilog
    assert "_ss_cnt <= _macro_boundary ? 1'd0 : (_ss_cnt + 1'd1);" in verilog
    assert "v_reg <= v_next;" in verilog
    assert "v_out <= v_next;" in verilog
    assert "if (_macro_boundary) begin" in verilog
    assert "&& !_thr_prev" in verilog
    assert "_ss_cnt <= 1'd0;" in verilog


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog is not installed")
@pytest.mark.parametrize("substeps", [1, 2])
def test_crossing_temporal_variants_are_valid_verilog(
    substeps: int,
    tmp_path: Path,
) -> None:
    """Icarus accepts both temporal branches as standalone synthesizable RTL."""
    source = tmp_path / f"sc_crossing_{substeps}.v"
    source.write_text(
        compile_to_verilog(
            _crossing_neuron(substeps=substeps),
            module_name=f"sc_crossing_{substeps}",
        ),
        encoding="utf-8",
    )

    subprocess.run(
        ["iverilog", "-g2012", "-t", "null", str(source)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
