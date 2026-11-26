# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SpiNNakerLIF: the engine takes the Python model's exact flow

"""The engine's SpiNNakerLIFNeuron takes the same exact-flow step as the Python model.

The Python model advances the membrane by the exact solution of
``tau_m dv/dt = -(v - v_rest) + I + i_offset`` under a constant input over one step and counts
the refractory period down to 0, never below. The engine took a forward-Euler step, so the
same public name gave different trajectories, and its refractory count could end below 0.
"""

from __future__ import annotations

from typing import Protocol, cast

import numpy as np
import pytest

from sc_neurocore.neurons.models.spinnaker_lif import SpiNNakerLIFNeuron
from tests.engine_requirement import require_engine

engine = require_engine()


class _EngineNeuron(Protocol):
    """The engine binding's stepping interface."""

    def step(self, current: float) -> int: ...

    def reset(self) -> None: ...

    def get_state(self) -> dict[str, float]: ...


def _pair() -> tuple[_EngineNeuron, SpiNNakerLIFNeuron]:
    """Return a fresh engine neuron and a fresh Python neuron, both at their defaults."""
    return cast(_EngineNeuron, engine.SpiNNakerLIFNeuron()), SpiNNakerLIFNeuron()


def _state(neuron: _EngineNeuron) -> tuple[float, float]:
    """Return the engine neuron's ``(v, refrac_count)``."""
    state = neuron.get_state()
    return state["v"], state["refrac_count"]


@pytest.mark.parametrize("current", [0.0, 19.0, 25.0, 80.0])
def test_a_constant_current_gives_the_same_trajectory(current: float) -> None:
    """For 2000 steps of a constant current, every spike, ``v`` and refractory count agree exactly."""
    rust, python = _pair()
    for step in range(2000):
        assert rust.step(current) == python.step(current), step
        assert _state(rust) == (python.v, python.refrac_count), step


def test_a_mixed_drive_gives_the_same_trajectory() -> None:
    """Over 2000 seeded sub- and suprathreshold inputs, spikes and state agree exactly."""
    rust, python = _pair()
    drive = np.random.default_rng(20140601).uniform(-10.0, 60.0, size=2000)
    spikes = 0
    for step, current in enumerate(drive.tolist()):
        fired = python.step(current)
        assert rust.step(current) == fired, step
        assert _state(rust) == (python.v, python.refrac_count), step
        spikes += fired
    assert spikes > 0


def test_the_first_spikes_come_at_the_same_steps() -> None:
    """Under 25.0 both spike first at steps 33 and 68."""
    rust, python = _pair()
    rust_spikes = [step for step in range(1, 101) if rust.step(25.0)]
    python_spikes = [step for step in range(1, 101) if python.step(25.0)]
    assert rust_spikes == python_spikes == [33, 68]


def test_reset_returns_both_to_rest() -> None:
    """After a spiking run, reset puts both back at rest and they agree again."""
    rust, python = _pair()
    for _ in range(300):
        rust.step(30.0)
        python.step(30.0)
    rust.reset()
    python.reset()
    assert _state(rust) == (python.v, python.refrac_count)
    assert rust.step(30.0) == python.step(30.0)
    assert _state(rust) == (python.v, python.refrac_count)
