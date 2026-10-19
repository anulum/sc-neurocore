# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (stochastic_lif_rejects) from former test_to_torch_bridge.py

"""StochasticLIF boundary of ``Network.to_torch()``: noise, refractoriness and entropy are refused."""

from __future__ import annotations

import pytest

from sc_neurocore.network import Network, Population
from sc_neurocore.neurons.stochastic_lif import StochasticLIFNeuron

torch = pytest.importorskip("torch")


def test_network_to_torch_rejects_stochastic_lif_with_noise() -> None:
    """Membrane noise has no deterministic bridge form and is refused."""
    pop = Population(
        StochasticLIFNeuron,
        1,
        params={"noise_std": 0.1, "refractory_period": 0, "v_reset": 0.0, "v_rest": 0.0},
    )
    net = Network(pop)

    with pytest.raises(NotImplementedError, match="noise_std == 0.0"):
        net.to_torch()


def test_network_to_torch_rejects_stochastic_lif_with_refractory_period() -> None:
    """A refractory period is not modelled by the bridge cell and is refused."""
    pop = Population(
        StochasticLIFNeuron,
        1,
        params={"noise_std": 0.0, "refractory_period": 1, "v_reset": 0.0, "v_rest": 0.0},
    )
    net = Network(pop)

    with pytest.raises(NotImplementedError, match="refractory_period == 0"):
        net.to_torch()


def test_network_to_torch_rejects_stochastic_lif_with_entropy_source() -> None:
    """An external entropy source is refused."""
    pop = Population(
        StochasticLIFNeuron,
        1,
        params={
            "noise_std": 0.0,
            "refractory_period": 0,
            "v_reset": 0.0,
            "v_rest": 0.0,
            "entropy_source": object(),
        },
    )
    net = Network(pop)

    with pytest.raises(NotImplementedError, match="external entropy_source"):
        net.to_torch()


def test_network_to_torch_accepts_stochastic_lif_when_reset_differs_from_rest() -> None:
    """The bridge hard-resets to v_reset, so v_reset != v_rest steps as the model does."""
    params = {"noise_std": 0.0, "refractory_period": 0, "v_reset": -1.0, "v_rest": 0.0}
    reference = Population(StochasticLIFNeuron, 1, params=params)
    expected = [reference.neurons[0].step(drive) for drive in (3.0, 0.0, 1.5)]
    pop = Population(StochasticLIFNeuron, 1, params=params)
    _counts, traces = Network(pop).to_torch()(
        torch.tensor([[[3.0]], [[0.0]], [[1.5]]], dtype=torch.float64), return_traces=True
    )
    assert traces[pop.label][:, 0, 0].tolist() == [float(e) for e in expected]
    # Reset to v_rest = 0 would fire at the third step; reset to -1 does not.
    assert expected == [1, 0, 0]
