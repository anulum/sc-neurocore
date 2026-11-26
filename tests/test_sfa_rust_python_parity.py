# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SFANeuron: the engine takes the Python model's coupled RK4 step

"""The engine's SFANeuron takes the same coupled RK4 step and spike rule as the Python model.

The Python model takes a candidate-first fourth-order Runge-Kutta step of the membrane
``tau_m dv/dt = -(v - v_rest) - g_sfa (v - e_k) + R I`` coupled with the adaptation
conductance ``dg_sfa/dt = -g_sfa / tau_sfa``. A spike resets ``v`` and adds ``delta_g`` to the
candidate conductance. The engine took a forward-Euler step of ``v`` and then decayed
``g_sfa`` by its exact exponential, so the same public name gave different trajectories.
"""

from __future__ import annotations

from typing import Protocol, cast

import numpy as np
import pytest

from sc_neurocore.neurons.models.sfa import SFANeuron
from tests.engine_requirement import require_engine

engine = require_engine()


class _EngineNeuron(Protocol):
    """The engine binding's stepping interface."""

    def step(self, current: float) -> int: ...

    def reset(self) -> None: ...

    def get_state(self) -> dict[str, float]: ...


def _pair() -> tuple[_EngineNeuron, SFANeuron]:
    """Return a fresh engine neuron and a fresh Python neuron, both at their defaults."""
    return cast(_EngineNeuron, engine.SFANeuron()), SFANeuron()


def _state(neuron: _EngineNeuron) -> tuple[float, float]:
    """Return the engine neuron's ``(v, g_sfa)``."""
    state = neuron.get_state()
    return state["v"], state["g_sfa"]


@pytest.mark.parametrize("current", [0.0, 2.5, 30.0, 60.0])
def test_a_constant_current_gives_the_same_trajectory(current: float) -> None:
    """For 2000 steps of a constant current, every spike, ``v`` and ``g_sfa`` agree exactly."""
    rust, python = _pair()
    for step in range(2000):
        assert rust.step(current) == python.step(current), step
        assert _state(rust) == (python.v, python.g_sfa), step


def test_a_mixed_drive_gives_the_same_trajectory() -> None:
    """Over 2000 seeded sub- and suprathreshold inputs, spikes and state agree exactly."""
    rust, python = _pair()
    drive = np.random.default_rng(20020601).uniform(-5.0, 60.0, size=2000)
    spikes = 0
    for step, current in enumerate(drive.tolist()):
        fired = python.step(current)
        assert rust.step(current) == fired, step
        assert _state(rust) == (python.v, python.g_sfa), step
        spikes += fired
    assert spikes > 0


def test_a_spike_adds_delta_g_to_the_candidate_conductance() -> None:
    """At the first spike both reset ``v`` and carry the decayed conductance plus ``delta_g``."""
    rust, python = _pair()
    spikes = [(rust.step(30.0), python.step(30.0)) for _ in range(11)]
    assert spikes[-1] == (1, 1) and all(pair == (0, 0) for pair in spikes[:-1])
    assert _state(rust) == (python.v, python.g_sfa) == (-70.0, 0.5)


def test_reset_returns_both_to_rest() -> None:
    """After a spiking run, reset restores ``v_rest`` and ``g_sfa = 0`` and they agree again."""
    rust, python = _pair()
    for _ in range(300):
        rust.step(30.0)
        python.step(30.0)
    rust.reset()
    python.reset()
    assert _state(rust) == (python.v, python.g_sfa) == (-70.0, 0.0)
    assert rust.step(30.0) == python.step(30.0)
    assert _state(rust) == (python.v, python.g_sfa)
