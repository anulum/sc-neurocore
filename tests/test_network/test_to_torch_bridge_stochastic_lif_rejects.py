# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (stochastic_lif_rejects) from former test_to_torch_bridge.py

from __future__ import annotations

from to_torch_bridge_support import *  # noqa: F403


def test_network_to_torch_rejects_stochastic_lif_with_noise() -> None:
    pop = Population(
        StochasticLIFNeuron,
        1,
        params={"noise_std": 0.1, "refractory_period": 0, "v_reset": 0.0, "v_rest": 0.0},
    )
    net = Network(pop)

    with pytest.raises(NotImplementedError, match="noise_std == 0.0"):
        net.to_torch()


def test_network_to_torch_rejects_stochastic_lif_with_refractory_period() -> None:
    pop = Population(
        StochasticLIFNeuron,
        1,
        params={"noise_std": 0.0, "refractory_period": 1, "v_reset": 0.0, "v_rest": 0.0},
    )
    net = Network(pop)

    with pytest.raises(NotImplementedError, match="refractory_period == 0"):
        net.to_torch()


def test_network_to_torch_rejects_stochastic_lif_with_entropy_source() -> None:
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


def test_network_to_torch_rejects_stochastic_lif_when_reset_differs_from_rest() -> None:
    pop = Population(
        StochasticLIFNeuron,
        1,
        params={"noise_std": 0.0, "refractory_period": 0, "v_reset": -1.0, "v_rest": 0.0},
    )
    net = Network(pop)

    with pytest.raises(NotImplementedError, match="v_reset == v_rest"):
        net.to_torch()
