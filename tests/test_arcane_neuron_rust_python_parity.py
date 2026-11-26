# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ArcaneNeuron: one deterministic step in Python and in the engine

"""The engine's ArcaneNeuron takes the Python model's step, and that step no longer depends on the CPU.

The Python model relaxes its three compartments exactly over each step and decides the
spike from the candidate fast state; the engine took Euler steps in a different order, so
the same public name gave different trajectories. The Python model also took the
predictor's norm through ``np.linalg.norm``, which goes through the BLAS dot kernel: a fused
multiply-add chain on CPUs that have it and plain products elsewhere, so its own trajectory
changed with the host. Both now take a plain left-to-right sum of squares and a plain mean
of the novelty history, and the engine follows the Python step bit for bit.
"""

from __future__ import annotations

from typing import Protocol, cast

import numpy as np
import pytest

from sc_neurocore.neurons.models.arcane_neuron import ArcaneNeuron
from tests.engine_requirement import require_engine

engine = require_engine()


class _EngineNeuron(Protocol):
    """The engine binding's stepping interface."""

    def step(self, current: float) -> int: ...

    def reset(self) -> None: ...

    def get_state(self) -> dict[str, float]: ...


def _pair() -> tuple[_EngineNeuron, ArcaneNeuron]:
    """Return a fresh engine neuron and a fresh Python neuron, both at their defaults."""
    return cast(_EngineNeuron, engine.ArcaneNeuron()), ArcaneNeuron()


def _state(neuron: _EngineNeuron) -> tuple[float, float, float]:
    """Return the engine neuron's ``(v_fast, v_work, v_deep)``."""
    state = neuron.get_state()
    return state["v_fast"], state["v_work"], state["v_deep"]


def _python_state(neuron: ArcaneNeuron) -> tuple[float, float, float]:
    """Return the Python neuron's ``(v_fast, v_work, v_deep)`` as floats."""
    return float(neuron.v_fast), float(neuron.v_work), float(neuron.v_deep)


@pytest.mark.parametrize("current", [0.0, 0.5, 2.0, 5.0])
def test_a_constant_current_gives_the_same_trajectory(current: float) -> None:
    """For 2000 steps of a constant current, every spike and all three compartments agree exactly."""
    rust, python = _pair()
    for step in range(2000):
        assert rust.step(current) == python.step(current), step
        assert _state(rust) == _python_state(python), step


def test_a_mixed_drive_gives_the_same_trajectory() -> None:
    """Over 2000 seeded inputs, spikes and the three compartments agree exactly."""
    rust, python = _pair()
    drive = np.random.default_rng(20261125).uniform(-1.0, 6.0, size=2000)
    spikes = 0
    for step, current in enumerate(drive.tolist()):
        fired = python.step(current)
        assert rust.step(current) == fired, step
        assert _state(rust) == _python_state(python), step
        spikes += fired
    assert spikes > 0


def test_the_predictor_norm_is_the_plain_sum_of_squares() -> None:
    """After learning, the Python predictor has the norm a left-to-right sum of squares gives."""
    neuron = ArcaneNeuron()
    for _ in range(300):
        neuron.step(2.0)
    w0, w1, w2 = (float(weight) for weight in neuron.w_pred)
    assert (w0, w1, w2) == (0.9161249065184631, 0.37682124738564304, 0.13682435154337574)


def test_reset_keeps_the_deep_compartment_in_both() -> None:
    """After a run, reset clears the fast and working compartments, keeps ``v_deep``, and they agree."""
    rust, python = _pair()
    for _ in range(500):
        rust.step(3.0)
        python.step(3.0)
    rust.reset()
    python.reset()
    assert _state(rust) == _python_state(python)
    assert _state(rust)[:2] == (0.0, 0.0)
    for _ in range(50):
        assert rust.step(3.0) == python.step(3.0)
        assert _state(rust) == _python_state(python)
