# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The engine's Loihi and Akida neurons follow the Python models step by step

"""Compare the engine's Loihi CUBA, Loihi 2 and Akida neurons with the Python models.

Loihi: both runtimes decay each state by its quotient truncated towards zero, so a negative state decays as its
positive mirror does. Lava's bit-accurate Loihi LIF model rounds its decayed state towards zero in the same
sign-magnitude way. The Python models floored the quotient, so a negative state leaked more than the same positive one.
For the drive -7, -3, 5, -9, -1, -4, the Python Loihi CUBA neuron ended at v = -33, u = -10 and the engine at
v = -48, u = -13.

Akida: a zero weight is not an input event. The weight factor of the next event is ``modulation`` multiplied in once
per event, and the accumulator keeps integrating after its one spike. The engine counted every call as an event and
stopped integrating after the spike, so the weights 3, 0, 3, 3 gave the Python model v = 6, rank 3 and the engine
v = 5, rank 4.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from sc_neurocore.neurons.models.akida_neuron import AkidaNeuron
from sc_neurocore.neurons.models.loihi2 import Loihi2Neuron
from sc_neurocore.neurons.models.loihi_cuba import LoihiCUBANeuron

SHORT_DRIVE = [-7, -3, 5, -9, -1, -4]


def _engine() -> Any:
    """Return the engine module, or skip when it is not built."""
    return pytest.importorskip("sc_neurocore_engine", reason="Rust engine not built")


def _seeded(seed: int, count: int, low: int, high: int, zero_share: float = 0.0) -> list[int]:
    """Return ``count`` seeded integers in ``[low, high]``, a share of them replaced by 0.

    Parameters
    ----------
    seed : int
        Seed of the generator.
    count : int
        How many values.
    low, high : int
        Inclusive bounds of the values.
    zero_share : float
        Probability that a value is 0 instead.
    """
    generator = random.Random(seed)
    return [
        0 if generator.random() < zero_share else generator.randint(low, high) for _ in range(count)
    ]


LOIHI_DRIVES = {
    "short negative": SHORT_DRIVE,
    "mixed sign": _seeded(20261115, 400, -300, 320),
    "constant 200": [200] * 60,
}

AKIDA_WEIGHTS = {
    "a zero between events": [3, 0, 3, 3],
    "events after the spike": [60, 0, 0, 50, 40, 0, 30, 25],
    "mixed sign with zeros": _seeded(1115, 80, -40, 120, zero_share=0.3),
}


def test_the_short_negative_drive_ends_where_truncation_puts_it() -> None:
    """The Python Loihi CUBA neuron ends the short drive at the engine's state, not at the floored one."""
    python = LoihiCUBANeuron()
    for value in SHORT_DRIVE:
        python.step(value)
    assert (python.v, python.u) == (-48, -13)


@pytest.mark.parametrize("drive", LOIHI_DRIVES.values(), ids=LOIHI_DRIVES.keys())
def test_loihi_cuba_follows_the_python_model_step_by_step(drive: list[int]) -> None:
    """Spikes, v and u agree on every step.

    Parameters
    ----------
    drive : list[int]
        The weighted input of each step.
    """
    engine = _engine().LoihiCUBANeuron()
    python = LoihiCUBANeuron()
    for step, value in enumerate(drive):
        assert engine.step(value) == python.step(value), f"spike differs at step {step}"
        state = engine.get_state()
        assert (state["v"], state["u"]) == (python.v, python.u), f"state differs at step {step}"


@pytest.mark.parametrize("drive", LOIHI_DRIVES.values(), ids=LOIHI_DRIVES.keys())
def test_loihi2_follows_the_python_model_step_by_step(drive: list[int]) -> None:
    """Spikes, s1, s2 and s3 agree on every step.

    Parameters
    ----------
    drive : list[int]
        The weighted input of each step.
    """
    engine = _engine().Loihi2Neuron()
    python = Loihi2Neuron()
    for step, value in enumerate(drive):
        assert engine.step(value) == python.step(value), f"spike differs at step {step}"
        state = engine.get_state()
        python_state = (python.s1, python.s2, python.s3)
        assert (state["s1"], state["s2"], state["s3"]) == python_state, (
            f"state differs at step {step}"
        )


def test_the_drives_reach_both_signs_and_spike() -> None:
    """The mixed drive takes both models below zero, and the mixed and constant drives make them fire."""
    for model in (LoihiCUBANeuron, Loihi2Neuron):
        lowest, spikes = 0, 0
        neuron = model()
        for value in LOIHI_DRIVES["mixed sign"]:
            spikes += neuron.step(value)
            lowest = min(lowest, neuron.u if isinstance(neuron, LoihiCUBANeuron) else neuron.s2)
        assert lowest < -100 and spikes > 0, model.__name__
        neuron = model()
        assert sum(neuron.step(value) for value in LOIHI_DRIVES["constant 200"]) > 1, model.__name__


@pytest.mark.parametrize("weights", AKIDA_WEIGHTS.values(), ids=AKIDA_WEIGHTS.keys())
def test_akida_follows_the_python_model_event_by_event(weights: list[int]) -> None:
    """Spikes, the accumulator and the rank agree after every event, through a reset and a second presentation.

    Parameters
    ----------
    weights : list[int]
        The weight of each input event; 0 is no event.
    """
    engine = _engine().AkidaNeuron(100)
    python = AkidaNeuron()
    for presentation in range(2):
        for event, weight in enumerate(weights):
            assert engine.step(weight) == python.step(weight), (
                f"spike differs at {presentation}/{event}"
            )
            state = engine.get_state()
            assert (state["v"], state["rank"]) == (python.v, python._rank), (
                f"state differs at {presentation}/{event}"
            )
        engine.reset()
        python.reset()


def test_the_akida_weights_cross_the_threshold_and_keep_integrating() -> None:
    """The second and third weight lists fire once, and the accumulator moves on after the spike."""
    for name in ("events after the spike", "mixed sign with zeros"):
        neuron = AkidaNeuron()
        spikes, after_spike = 0, []
        for weight in AKIDA_WEIGHTS[name]:
            spikes += neuron.step(weight)
            if neuron._spiked:
                after_spike.append(neuron.v)
        assert spikes == 1, name
        assert len(set(after_spike)) > 1, name
