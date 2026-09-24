# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio catalogue profile templates for hardware lowering

"""The Studio profile templates compute the catalogue models, not NIR's versions of them."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sc_neurocore.ir.scnir_convert import _population_encoding
from sc_neurocore.neurons.models.lapicque import SCLapicqueLIFNeuron
from sc_neurocore.neurons.models.perfect_integrator import PerfectIntegratorNeuron
from sc_neurocore.nir_bridge.fpga_compiler import compile_network_to_fpga
from sc_neurocore.nir_bridge.fpga_neuron_rtl import _population_neuron
from sc_neurocore.nir_bridge.neuron_graph_contracts import (
    ConnectionSpec,
    NeuronGraph,
    NeuronSpec,
)


def _spec(neuron_type: str, params: dict[str, float], dt: float = 1.0) -> NeuronSpec:
    return NeuronSpec(
        name="pop",
        neuron_type=neuron_type,
        n_neurons=1,
        params={name: np.array([value]) for name, value in params.items()},
        dt=dt,
    )


def _run(neuron, drive: list[float]) -> tuple[list[float], list[int]]:
    trace, spikes = [], []
    for current in drive:
        spikes.append(int(neuron.step(current)))
        trace.append(float(neuron.state["v"] if hasattr(neuron, "state") else neuron.v))
    return trace, spikes


def test_sc_lif_is_the_exact_step_of_the_catalogue_model() -> None:
    tau, dt, resistance, v_rest = 12.0, 0.5, 2.0, 0.1
    decay = math.exp(-dt / tau)
    template = _population_neuron(
        "sc_lif",
        _spec(
            "sc_lif",
            {
                "decay": decay,
                "gain": 1.0 - decay,
                "v_rest": v_rest,
                "r": resistance,
                "v_threshold": 1.0,
                "v_reset": 0.0,
            },
            dt,
        ),
    )
    model = SCLapicqueLIFNeuron(
        v=v_rest, v_rest=v_rest, v_reset=0.0, v_threshold=1.0, dt=dt, tau=tau, resistance=resistance
    )
    drive = [0.9] * 60 + [0.0] * 10 + [0.8] * 60

    template_trace, template_spikes = _run(template, drive)
    model_trace = []
    model_spikes = []
    for current in drive:
        model_spikes.append(model.step(current))
        model_trace.append(model.v)

    assert template_spikes == model_spikes
    assert sum(model_spikes) >= 4
    np.testing.assert_allclose(template_trace, model_trace, rtol=0.0, atol=1e-12)


def test_sc_if_fires_at_the_threshold_itself_as_the_catalogue_model_does() -> None:
    """0.25 four times reaches 1.0 exactly: sc_inclusive fires, NIR's ``if`` does not."""
    params = {"r": 1.0, "v_threshold": 1.0, "v_reset": 0.0}
    drive = [0.25] * 8
    model = PerfectIntegratorNeuron(v=0.0, c_m=1.0, v_threshold=1.0, v_reset=0.0, dt=1.0)
    model_spikes = [model.step(current) for current in drive]

    _trace, sc_spikes = _run(_population_neuron("sc_if", _spec("sc_if", params)), drive)
    _trace, nir_spikes = _run(_population_neuron("if", _spec("if", params)), drive)

    assert sc_spikes == model_spikes == [0, 0, 0, 1, 0, 0, 0, 1]
    assert nir_spikes != model_spikes


@pytest.mark.parametrize("neuron_type", ["sc_lif", "sc_if"])
def test_the_studio_profiles_compile_to_their_own_modules(neuron_type: str) -> None:
    params = (
        {"decay": 0.9, "gain": 0.1, "v_rest": 0.0, "r": 1.0, "v_threshold": 1.0, "v_reset": 0.0}
        if neuron_type == "sc_lif"
        else {"r": 1.0, "v_threshold": 1.0, "v_reset": 0.0}
    )
    spec = NeuronSpec(
        name="pop",
        neuron_type=neuron_type,
        n_neurons=2,
        params={name: np.full(2, value) for name, value in params.items()},
        dt=1.0,
    )
    graph = NeuronGraph(
        populations=[spec],
        connections=[
            ConnectionSpec(src="drive", dst="pop", weights=np.eye(2), bias=None, delay_steps=0)
        ],
        input_pop="drive",
        output_pop="pop",
        dt=1.0,
    )

    result = compile_network_to_fpga(graph, module_name="studio_profile", interconnect="direct")

    assert list(result.neuron_modules) == [neuron_type]
    module = result.neuron_modules[neuron_type]
    assert f"module sc_nir_{neuron_type}" in module
    assert ">=" in module
    assert _population_encoding(neuron_type) == "unipolar"
