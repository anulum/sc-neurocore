# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — negative graded values and voltages on the folded buses stay negative

"""The folded interconnect reads signed words from ``val_bus`` and ``v_bus``.

A Verilog part-select is unsigned, so a negative graded spike value or a negative
analogue voltage read from a packed bus once entered the product and the source
threshold comparison as a large positive number (RIGOR NEURO-GRADED-FOLDED-SIGNED-01:
a value of -0.5 times a weight of +1 gave +32767 instead of -128). Each case below is
built so that the sign decides whether the targets fire: the direct path (whose
values arrive on signed ports) must show the expected firing or silence, and the
folded raster must equal it, with and without a delay. Without Icarus the cases skip.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.folded_interconnect_support import (
    _HAVE_IVERILOG,
    ConnectionSpec,
    NeuronGraph,
    NeuronSpec,
    _parity_rasters,
)

pytestmark = pytest.mark.skipif(not _HAVE_IVERILOG, reason="iverilog/vvp not installed")

_NEGATIVE_GRADED = {
    "equations": ["dv/dt = -v / tau + I"],
    "threshold": "v > 1.0",
    "reset": "v = 0.0",
    "spike_value": "-0.75 * v",
    "default_params": {"tau": 4.0},
}
_TARGET = {
    "equations": ["dv/dt = -v / tau + I"],
    "threshold": "v > 0.3",
    "reset": "v = 0.0",
    "default_params": {"tau": 6.0},
}


def _graded_graph(weight: float, delay: int, threshold: float | None) -> NeuronGraph:
    """Return stim → three negative-valued graded sources → two targets at ``weight``."""
    src = NeuronSpec(
        name="src", neuron_type="src_t", n_neurons=3, dt=0.25, template=_NEGATIVE_GRADED
    )
    dst = NeuronSpec(name="dst", neuron_type="dst_t", n_neurons=2, dt=0.25, template=_TARGET)
    conns = [
        ConnectionSpec(src="stim", dst="src", weights=np.array([[1.0], [0.8], [0.6]], np.float32)),
        ConnectionSpec(
            src="src",
            dst="dst",
            weights=np.full((2, 3), weight, dtype=np.float32),
            delay_steps=delay,
            source_threshold=None if threshold is None else np.full(3, threshold, np.float32),
        ),
    ]
    return NeuronGraph(
        populations=[src, dst], connections=conns, input_pop="stim", output_pop="dst", dt=0.25
    )


def _targets_fire(raster: list[str]) -> bool:
    """Return whether any target (the raster's first two columns) fired."""
    return any("1" in row[:2] for row in raster)


@pytest.mark.parametrize("delay", [0, 2], ids=["now", "delayed"])
@pytest.mark.parametrize(
    ("weight", "threshold", "fires"),
    [
        (-0.6, None, True),  # negative value · negative weight: excitatory
        (0.6, None, False),  # negative value · positive weight: inhibitory
        (0.6, 0.25, False),  # a negative value is below a positive source threshold
        (0.6, -2.0, True),  # and above a lower negative one: the weight passes
    ],
    ids=["neg-neg", "neg-pos", "below-threshold", "above-threshold"],
)
def test_negative_graded_values_keep_their_sign(
    weight: float, threshold: float | None, fires: bool, delay: int
) -> None:
    """Direct fires exactly when the signed arithmetic says so; folded equals direct."""
    direct, folded = _parity_rasters(_graded_graph(weight, delay, threshold), [1.5], 5)
    assert _targets_fire(direct) is fires
    assert any("1" in row[2:] for row in direct), "the graded sources spike"
    assert folded == direct


def _analogue_graph(weight: float, delay: int) -> NeuronGraph:
    """Return an li population (driven negative by the stimulus) feeding three lif neurons."""
    li = NeuronSpec(name="a", neuron_type="li", n_neurons=3, params={}, dt=1.0)
    lif = NeuronSpec(name="b", neuron_type="lif", n_neurons=3, params={}, dt=1.0)
    return NeuronGraph(
        populations=[li, lif],
        connections=[
            ConnectionSpec(
                src="a",
                dst="b",
                weights=np.full((3, 3), weight, dtype=np.float32),
                delay_steps=delay,
            )
        ],
        input_pop="a",
        output_pop="b",
        dt=1.0,
    )


@pytest.mark.parametrize("delay", [0, 2], ids=["now", "delayed"])
@pytest.mark.parametrize(("weight", "fires"), [(-0.5, True), (0.5, False)], ids=["neg", "pos"])
def test_negative_analogue_voltages_keep_their_sign(weight: float, fires: bool, delay: int) -> None:
    """A negative li voltage times a negative weight excites; times a positive one inhibits."""
    direct, folded = _parity_rasters(_analogue_graph(weight, delay), [-4.0, -3.5, -3.0], 6)
    assert any("1" in row for row in direct) is fires
    assert folded == direct
