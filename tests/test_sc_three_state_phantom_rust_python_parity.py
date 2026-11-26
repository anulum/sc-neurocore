# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC three-state phantom burster: the engine takes the Python model's RK4 step

"""The engine's SCThreeStatePhantomBurster takes the same RK4 step as the Python model.

The Python model takes a fourth-order Runge-Kutta step of the voltage and the two slow
potassium gates, with steady-state activations from a sign-split Boltzmann function, and
clamps the slow gates of the candidate to [0, 1]. The engine took a forward-Euler step with
the direct logistic and no clamp, so the same public name gave different trajectories.
"""

from __future__ import annotations

from typing import Protocol, cast

import numpy as np
import pytest

from sc_neurocore.neurons.models.sc_three_state_phantom import SCThreeStatePhantomBurster
from tests.engine_requirement import require_engine

engine = require_engine()


class _EngineNeuron(Protocol):
    """The engine binding's stepping interface."""

    def step(self, current: float) -> int: ...

    def reset(self) -> None: ...

    def get_state(self) -> dict[str, float]: ...


def _pair() -> tuple[_EngineNeuron, SCThreeStatePhantomBurster]:
    """Return a fresh engine burster and a fresh Python burster, both at their defaults."""
    return cast(_EngineNeuron, engine.SCThreeStatePhantomBurster()), SCThreeStatePhantomBurster()


def _state(neuron: _EngineNeuron) -> tuple[float, float, float]:
    """Return the engine burster's ``(v, s1, s2)``."""
    state = neuron.get_state()
    return state["v"], state["s1"], state["s2"]


@pytest.mark.parametrize("current", [0.0, 20.0, 80.0, 200.0])
def test_a_constant_current_gives_the_same_trajectory(current: float) -> None:
    """For 2000 steps of a constant current, every crossing, ``v``, ``s1`` and ``s2`` agree exactly."""
    rust, python = _pair()
    for step in range(2000):
        assert rust.step(current) == python.step(current), step
        assert _state(rust) == (python.v, python.s1, python.s2), step


def test_a_mixed_drive_gives_the_same_trajectory() -> None:
    """Over 2000 seeded inputs, crossings and the three state variables agree exactly."""
    rust, python = _pair()
    drive = np.random.default_rng(20110601).uniform(-10.0, 30.0, size=2000)
    for step, current in enumerate(drive.tolist()):
        assert rust.step(current) == python.step(current), step
        assert _state(rust) == (python.v, python.s1, python.s2), step


def test_a_strong_current_crosses_the_threshold_at_the_same_step() -> None:
    """Under 200.0 both report their first upward crossing at the same step."""
    rust, python = _pair()
    rust_steps = [step for step in range(4000) if rust.step(200.0)]
    python_steps = [step for step in range(4000) if python.step(200.0)]
    assert rust_steps == python_steps
    assert rust_steps


def test_reset_returns_both_to_the_initial_state() -> None:
    """After a run, reset restores ``(-50, 0.1, 0.1)`` and they agree again."""
    rust, python = _pair()
    for _ in range(500):
        rust.step(80.0)
        python.step(80.0)
    rust.reset()
    python.reset()
    assert _state(rust) == (python.v, python.s1, python.s2) == (-50.0, 0.1, 0.1)
    assert rust.step(80.0) == python.step(80.0)
    assert _state(rust) == (python.v, python.s1, python.s2)
