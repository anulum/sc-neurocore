# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — a stochastic threshold fires as often in Q1.f RTL as in the model

"""The LFSR stochastic threshold compares the probability with 1.0 at every format.

At Q1.15 the value 1.0 is 32768, which a 16-bit signed literal wraps to -32768:
every probability then compared ``>= 1.0`` and the neuron fired on every step
(RIGOR NEURO-POISSON-Q1-01: 32 spikes in 32 steps at p = 0.5, against 19 in the
model). The comparisons with 1.0 now take one bit more when 1.0 needs it. Each
case runs the seeded model and the registered RTL (Icarus) for 64 steps and the
two event rasters must be equal; the endpoints 0 and (where representable) 1 and
narrow and wide formats are included. Without Icarus the cases skip.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from sc_neurocore.compiler.verilog_compiler import compile_to_verilog
from sc_neurocore.neurons.equation_builder import EquationNeuron
from tests.bit_true_cosim_support import _rows, _testbench
from tests.cosim_support import HAS_IVERILOG

_STEPS = 64


def _neuron(probability: float) -> EquationNeuron:
    """Return a stateless Poisson neuron firing with ``probability`` per step."""
    return EquationNeuron(
        equations={"v": "0.0"},
        state={"v": 0.0},
        parameters={"p": probability},
        dt=1.0,
        method="map",
        detection="poisson",
        probability_expression="p",
        rng_seed=0xACE1,
    )


_FORMATS = [(16, 15), (16, 14), (16, 8), (32, 31), (32, 16)]
_CASES = [
    (probability, data_width, fraction)
    for data_width, fraction in _FORMATS
    for probability in (0.0, 0.25, 0.5, 0.875, 1.0)
    # 1.0 itself is outside a Q1.f word; every other case fits every format.
    if probability < 1 << (data_width - fraction - 1)
]


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
@pytest.mark.parametrize(("probability", "data_width", "fraction"), _CASES)
def test_rtl_events_equal_the_model(probability: float, data_width: int, fraction: int) -> None:
    """Sixty-four seeded steps: the RTL raster equals the model's, endpoints included."""
    model = _neuron(probability)
    expected = [model.step(0.0) for _ in range(_STEPS)]
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        (tmp / "m.v").write_text(
            compile_to_verilog(_neuron(probability), "m", data_width, fraction)
        )
        (tmp / "tb.v").write_text(
            _testbench(_neuron(probability), "m", 0.0, _STEPS, data_width, fraction)
        )
        subprocess.run(
            ["iverilog", "-g2012", "-o", str(tmp / "s"), str(tmp / "m.v"), str(tmp / "tb.v")],
            check=True,
            capture_output=True,
            timeout=60,
        )
        out = subprocess.run(["vvp", str(tmp / "s")], capture_output=True, text=True, timeout=60)
    events = [int(row[0]) for row in _rows(out.stdout)]
    assert events == expected
    if 0.0 < probability < 1.0:
        assert 0 < sum(expected) < _STEPS
