# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NonlinearLIF: the engine takes the Python model's RK4 step

"""The engine's NonlinearLIFNeuron takes the same RK4 step and spike rule as the Python model.

The Python model, the Go and Julia lanes and the safety crate take a candidate-first
fourth-order Runge-Kutta step of ``c_m dv/dt = a (v - v_rest)(v - v_crit) - w + I`` and
``tau_w dw/dt = b (v - v_rest) - w``, and spike whenever the candidate ``v`` reaches
``v_threshold``. The engine took a semi-implicit Euler step and spiked only on an upward
crossing, so the same public name gave different trajectories (the 2026-11-24 audit measured
``v`` = -63.388 against -63.340).
"""

from __future__ import annotations

from typing import Protocol, cast

import numpy as np
import pytest

from sc_neurocore.neurons.models.nlif import NonlinearLIFNeuron
from tests.engine_requirement import require_engine

engine = require_engine()


class _EngineNeuron(Protocol):
    """The engine binding's stepping interface."""

    def step(self, current: float) -> int: ...

    def reset(self) -> None: ...

    def get_state(self) -> dict[str, float]: ...


def _pair() -> tuple[_EngineNeuron, NonlinearLIFNeuron]:
    """Return a fresh engine neuron and a fresh Python neuron, both at their defaults."""
    return cast(_EngineNeuron, engine.NonlinearLIFNeuron()), NonlinearLIFNeuron()


def _state(neuron: _EngineNeuron) -> tuple[float, float]:
    """Return the engine neuron's ``(v, w)``."""
    state = neuron.get_state()
    return state["v"], state["w"]


@pytest.mark.parametrize("current", [0.0, 2.5, 10.0, 500.0])
def test_a_constant_current_gives_the_same_trajectory(current: float) -> None:
    """For 2000 steps of a constant current, every spike, ``v`` and ``w`` agree exactly."""
    rust, python = _pair()
    for step in range(2000):
        assert rust.step(current) == python.step(current), step
        assert _state(rust) == (python.v, python.w), step


def test_a_mixed_drive_gives_the_same_trajectory() -> None:
    """Over 2000 seeded sub- and suprathreshold inputs, spikes and state agree exactly."""
    rust, python = _pair()
    drive = np.random.default_rng(20080601).uniform(-5.0, 40.0, size=2000)
    spikes = 0
    for step, current in enumerate(drive.tolist()):
        fired = python.step(current)
        assert rust.step(current) == fired, step
        assert _state(rust) == (python.v, python.w), step
        spikes += fired
    assert spikes > 0


def test_a_candidate_above_threshold_spikes_without_an_upward_crossing() -> None:
    """A candidate at or above ``v_threshold`` spikes in both, whatever the previous ``v`` was."""
    rust, python = _pair()
    first = [rust.step(1e4), python.step(1e4)]
    assert first == [1, 1]
    assert _state(rust) == (python.v, python.w)
    assert [rust.step(1e4), python.step(1e4)] == [1, 1]


def test_reset_returns_both_to_rest() -> None:
    """After a spiking run, reset restores ``v_rest`` and ``w = 0`` and they agree again."""
    rust, python = _pair()
    for _ in range(300):
        rust.step(30.0)
        python.step(30.0)
    rust.reset()
    python.reset()
    assert _state(rust) == (python.v, python.w) == (-65.0, 0.0)
    assert rust.step(30.0) == python.step(30.0)
    assert _state(rust) == (python.v, python.w)
