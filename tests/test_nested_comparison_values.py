# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — a comparison compared again is the number 1.0 or 0.0 on every target

"""A comparison used as a comparison operand is a number, as in the Python model.

``(v > 0.25) > 0.5`` is ``True > 0.5`` in the model. The RTL and the C kernel
compared the one-bit truth value with the encoded 0.5 (128 at Q8.8), so the
threshold never fired, and the Rust kernel did not compile (bool against i64)
(RIGOR NEURO-COMPILER-NESTED-BOOL-01). A comparison operand is now promoted to
1.0 or 0.0 in the Q-format, like a comparison used in arithmetic. Each case is
run in Python and through the RTL (Icarus), the C kernel and the Rust kernel;
the four traces of spike and state must agree word for word. Simple and chained
comparisons are the healthy controls. Without Icarus/gcc the cases skip; without
rustc the Rust trace is left out.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from sc_neurocore.neurons.equation_builder import EquationNeuron, from_equations
from tests.bit_true_cosim_support import HAS_COSIM, HAS_RUST, _c_trace, _rust_trace, _verilog_trace

_DW, _FR, _STEPS = 16, 8, 8

_CASES = {
    "nested-threshold": ("dv/dt = 0.75", "(v > 0.25) > 0.5", [1] * _STEPS),
    "simple-threshold": ("dv/dt = 0.75", "v > 0.25", [1] * _STEPS),
    "nested-arithmetic": ("dv/dt = 0.5 - ((v > 0.25) > 0.5) * 0.375", "v > 0.25", [1, 0] * 4),
    "nested-in-chain": ("dv/dt = 0.75", "0.5 < (v > 0.25) <= 1.0", [1] * _STEPS),
    "simple-chain": ("dv/dt = 0.75", "0.5 < v < 1.0", [1] * _STEPS),
}


def _neuron(equation: str, threshold: str) -> EquationNeuron:
    """Return the map neuron of ``equation`` with ``threshold`` (reset leaves v unchanged)."""
    return from_equations(
        equation, threshold=threshold, reset="v = v", init={"v": 0.0}, dt=1.0, method="map"
    )


def _python_trace(neuron: EquationNeuron) -> list[list[str]]:
    """Run the model and return rows of spike and the state's code at Q8.8."""
    rows = []
    for _ in range(_STEPS):
        spike = neuron.step(0.0)
        rows.append([str(int(spike)), str(round(neuron.state["v"] * (1 << _FR)))])
    return rows


@pytest.mark.skipif(not HAS_COSIM, reason="Icarus Verilog / gcc not available")
@pytest.mark.parametrize("case", sorted(_CASES))
def test_python_rtl_c_and_rust_agree(case: str) -> None:
    """Spikes and states agree on all four targets and match the stated raster."""
    equation, threshold, spikes = _CASES[case]
    python = _python_trace(_neuron(equation, threshold))
    assert [int(row[0]) for row in python] == spikes
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rtl = _verilog_trace(_neuron(equation, threshold), "m", 0.0, _STEPS, _DW, _FR, tmp)
        c = _c_trace(_neuron(equation, threshold), "m", 0, _STEPS, _DW, _FR, tmp)
        rust = (
            _rust_trace(_neuron(equation, threshold), "m", 0, _STEPS, _DW, _FR, tmp)
            if HAS_RUST
            else c
        )
    assert rtl == c == rust == python
