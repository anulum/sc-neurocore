# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Sherman-Rinzel-Keizer: the engine takes the Python model's RK4 step

"""The engine's ShermanRinzelKeizerNeuron takes the same RK4 step as the Python model.

The Python model of the Sherman, Rinzel & Keizer (1988) burster takes a fourth-order
Runge-Kutta step of the voltage, the potassium gate ``n`` and the slow gate ``s``, with the
steady-state sigmoids' arguments clamped to [-80, 80], and reports an upward crossing of
``v_threshold``. The engine took a forward-Euler step with unclamped sigmoids, so the same
public name gave different bursts.
"""

from __future__ import annotations

from typing import Protocol, cast

import numpy as np
import pytest

from sc_neurocore.neurons.models.sherman_rinzel_keizer import ShermanRinzelKeizerNeuron
from tests.engine_requirement import require_engine

engine = require_engine()


class _EngineNeuron(Protocol):
    """The engine binding's stepping interface."""

    def step(self, current: float) -> int: ...

    def reset(self) -> None: ...

    def get_state(self) -> dict[str, float]: ...


def _pair() -> tuple[_EngineNeuron, ShermanRinzelKeizerNeuron]:
    """Return a fresh engine neuron and a fresh Python neuron, both at their defaults."""
    return cast(_EngineNeuron, engine.ShermanRinzelKeizerNeuron()), ShermanRinzelKeizerNeuron()


def _state(neuron: _EngineNeuron) -> tuple[float, float, float]:
    """Return the engine neuron's ``(v, n, s)``."""
    state = neuron.get_state()
    return state["v"], state["n"], state["s"]


@pytest.mark.parametrize("current", [0.0, 5.0, 20.0, 60.0])
def test_a_constant_current_gives_the_same_trajectory(current: float) -> None:
    """For 2000 steps of a constant current, every crossing, ``v``, ``n`` and ``s`` agree exactly."""
    rust, python = _pair()
    for step in range(2000):
        assert rust.step(current) == python.step(current), step
        assert _state(rust) == (python.v, python.n, python.s), step


def test_a_mixed_drive_gives_the_same_trajectory() -> None:
    """Over 2000 seeded inputs, crossings and the three state variables agree exactly."""
    rust, python = _pair()
    drive = np.random.default_rng(19880601).uniform(-10.0, 30.0, size=2000)
    crossings = 0
    for step, current in enumerate(drive.tolist()):
        fired = python.step(current)
        assert rust.step(current) == fired, step
        assert _state(rust) == (python.v, python.n, python.s), step
        crossings += fired
    assert crossings > 0


def test_reset_returns_both_to_the_initial_state() -> None:
    """After a bursting run, reset restores ``(-50, 0.1, 0.1)`` and they agree again."""
    rust, python = _pair()
    for _ in range(500):
        rust.step(5.0)
        python.step(5.0)
    rust.reset()
    python.reset()
    assert _state(rust) == (python.v, python.n, python.s) == (-50.0, 0.1, 0.1)
    assert rust.step(5.0) == python.step(5.0)
    assert _state(rust) == (python.v, python.n, python.s)
