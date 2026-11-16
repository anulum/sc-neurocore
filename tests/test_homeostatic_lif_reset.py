# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The homeostatic LIF's reset returns every state it adapts

"""``reset_state`` returns the homeostatic LIF to the state it was built with.

The base contract asks a reset for every internal state's initial value.
"""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.homeostatic_lif import HomeostaticLIFNeuron


def _adapted(neuron: HomeostaticLIFNeuron, steps: int = 300) -> HomeostaticLIFNeuron:
    """Drive the neuron hard enough that its threshold and rate trace move.

    Parameters
    ----------
    neuron : HomeostaticLIFNeuron
        The neuron to drive.
    steps : int
        Number of steps at input 1.5.

    Returns
    -------
    HomeostaticLIFNeuron
        The same neuron, adapted.
    """
    for _ in range(steps):
        neuron.step(1.5)
    return neuron


@pytest.mark.parametrize("rate_trace", [0.0, 0.3])
def test_reset_restores_the_threshold_and_the_rate_trace_it_was_built_with(
    rate_trace: float,
) -> None:
    """After adaptation, ``reset_state`` restores ``v_threshold``, ``rate_trace`` and ``v``.

    Parameters
    ----------
    rate_trace : float
        The rate trace the neuron is built with.
    """
    neuron = HomeostaticLIFNeuron(
        target_rate=0.1, noise_std=0.0, rate_trace=rate_trace, v_threshold=1.2
    )
    _adapted(neuron)
    assert neuron.v_threshold != pytest.approx(1.2)
    assert neuron.rate_trace != pytest.approx(rate_trace)
    neuron.reset_state()
    state = neuron.get_state()
    assert state["v"] == neuron.v_rest
    assert state["refractory"] == 0
    assert state["threshold"] == 1.2
    assert state["rate_trace"] == rate_trace


def test_a_reset_neuron_repeats_its_first_run() -> None:
    """With no noise, the spikes after a reset equal the spikes of a fresh neuron."""
    fresh = HomeostaticLIFNeuron(target_rate=0.1, noise_std=0.0)
    first = [fresh.step(1.5) for _ in range(200)]
    fresh.reset_state()
    again = [fresh.step(1.5) for _ in range(200)]
    assert again == first
