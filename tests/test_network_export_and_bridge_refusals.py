# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Verilog export and Torch bridge refuse what they cannot reproduce

"""A population the RTL or the Torch cell cannot reproduce is refused, not approximated."""

from __future__ import annotations

from pathlib import Path

import pytest

from sc_neurocore.exceptions import SCHardwareError
from sc_neurocore.network import Network, Population
from sc_neurocore.network.export import export_verilog


def test_export_refuses_neurons_with_different_thresholds(tmp_path: Path) -> None:
    """One exported module steps every neuron, so a per-neuron threshold is refused."""
    population = Population("SCLapicqueLIFNeuron", 3, label="mixed")
    population.neurons[2].v_threshold += 0.5
    with pytest.raises(SCHardwareError, match="export needs one v_threshold for every neuron"):
        export_verilog(Network(population), str(tmp_path / "out"))
    assert not (tmp_path / "out" / "sc_network_top.v").exists()


def test_export_refuses_an_external_entropy_source(tmp_path: Path) -> None:
    """A noiseless StochasticLIF neuron with an external entropy source is not exported."""
    population = Population(
        "StochasticLIFNeuron", 2, params={"noise_std": 0.0, "refractory_period": 0}, label="ext"
    )
    population.neurons[1].entropy_source = object()
    with pytest.raises(SCHardwareError, match="an external entropy_source is not exported"):
        export_verilog(Network(population), str(tmp_path / "out"))


def test_torch_flow_cell_refuses_a_voltage_at_threshold() -> None:
    """The SC LIF Torch cell refuses ``v >= v_threshold`` with the Python model's own message."""
    torch = pytest.importorskip("torch")
    population = Population("SCLapicqueLIFNeuron", 2, label="flow")
    population.neurons[0].v = population.neurons[0].v_threshold
    with pytest.raises(ValueError) as python_error:
        population.neurons[0].step(0.0)
    bridge = Network(population).to_torch()
    with pytest.raises(ValueError) as torch_error:
        bridge(torch.zeros((1, 1, 2), dtype=torch.float32))
    assert str(torch_error.value) == str(python_error.value)
    assert "v must be below v_threshold for the SC LIF profile" in str(torch_error.value)
