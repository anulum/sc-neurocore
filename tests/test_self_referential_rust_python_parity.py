# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SelfReferentialNeuron: one warm-up rule in the engine and in Python

"""The engine's SelfReferentialNeuron follows the Python model step for step, and Python honours ``window``.

The rate behind the adaptive time constant is the number of spikes in the last ``window``
steps divided by the steps seen since the last reset, at most ``window``. The Python model
took the rate over the steps seen so far; the engine divided by the full 50-step window from
the first step, so the two parted at the step after the first spike (step 8 under a constant
2.0). The Python history also held 50 steps whatever ``window`` said.
"""

from __future__ import annotations

from typing import Protocol, cast

import numpy as np

from sc_neurocore.neurons.models.ai_optimized import SelfReferentialNeuron
from tests.engine_requirement import require_engine

engine = require_engine()


class _EngineNeuron(Protocol):
    """The engine binding's stepping interface."""

    def step(self, current: float) -> int: ...

    def reset(self) -> None: ...

    def get_state(self) -> dict[str, float]: ...


def _pair() -> tuple[_EngineNeuron, SelfReferentialNeuron]:
    """Return a fresh engine neuron and a fresh Python neuron, both at their defaults."""
    return cast(_EngineNeuron, engine.SelfReferentialNeuron()), SelfReferentialNeuron()


def test_the_step_after_the_first_spike_uses_the_steps_seen() -> None:
    """Under 2.0 both spike first at step 7, and at step 8 both use the rate 1/7."""
    rust, python = _pair()
    for step in range(1, 8):
        assert rust.step(2.0) == python.step(2.0) == (1 if step == 7 else 0)
    rust.step(2.0)
    python.step(2.0)
    tau_eff = 10.0 * (1.0 + (1.0 / 7.0) / 0.1)
    assert rust.get_state()["v"] == python.v == 2.0 / tau_eff


def test_every_step_of_a_mixed_drive_matches_the_python_model() -> None:
    """Over 2000 seeded inputs, through the warm-up and well past it, spikes and ``v`` agree exactly."""
    rust, python = _pair()
    drive = np.random.default_rng(20261125).uniform(-0.5, 3.0, size=2000)
    spikes = 0
    for step, current in enumerate(drive.tolist()):
        fired = python.step(current)
        assert rust.step(current) == fired, step
        assert rust.get_state()["v"] == python.v, step
        spikes += fired
    assert spikes > 50


def test_reset_restarts_the_warm_up_in_both() -> None:
    """After reset both count steps from zero again and agree on the next spike train."""
    rust, python = _pair()
    for _ in range(120):
        rust.step(2.0)
        python.step(2.0)
    rust.reset()
    python.reset()
    assert python._step_count == 0 and len(python._history) == 0
    assert [rust.step(2.0) for _ in range(30)] == [python.step(2.0) for _ in range(30)]


def test_python_keeps_window_steps_of_history() -> None:
    """A window of 10 keeps 10 steps, and the rate after them is spikes over 10, not over 50."""
    neuron = SelfReferentialNeuron(window=10)
    assert neuron._history.maxlen == 10
    for _ in range(40):
        neuron.step(2.0)
    assert len(neuron._history) == 10
    wide = SelfReferentialNeuron(window=50)
    for _ in range(40):
        wide.step(2.0)
    narrow_rate = sum(neuron._history) / 10
    wide_rate = sum(wide._history) / 40
    v_narrow, v_wide = neuron.v, wide.v
    expected_narrow = v_narrow + (-v_narrow + 2.0) / (10.0 * (1.0 + narrow_rate / 0.1))
    expected_wide = v_wide + (-v_wide + 2.0) / (10.0 * (1.0 + wide_rate / 0.1))
    assert narrow_rate != wide_rate
    assert neuron.step(2.0) == (1 if expected_narrow >= 1.0 else 0)
    assert wide.step(2.0) == (1 if expected_wide >= 1.0 else 0)
    assert neuron.v == (0.0 if expected_narrow >= 1.0 else expected_narrow)
    assert wide.v == (0.0 if expected_wide >= 1.0 else expected_wide)


def test_a_window_changed_after_construction_resizes_the_history() -> None:
    """Setting ``window`` on a live neuron takes effect at the next step, keeping the latest entries."""
    neuron = SelfReferentialNeuron()
    for _ in range(30):
        neuron.step(2.0)
    latest = list(neuron._history)[-5:]
    neuron.window = 5
    neuron.step(2.0)
    assert neuron._history.maxlen == 5
    assert list(neuron._history)[:4] == latest[1:]


def _rate_after_live_change(new_window: int) -> tuple[SelfReferentialNeuron, float]:
    """Run 80 steps of 2.0, change ``window`` and return the neuron and the expected rate.

    Parameters
    ----------
    new_window : int
        Window set on the live neuron after the 80 steps.

    Returns
    -------
    tuple[SelfReferentialNeuron, float]
        The neuron and the spikes over the entries it holds within the new window.
    """
    neuron = SelfReferentialNeuron()
    for _ in range(80):
        neuron.step(2.0)
    neuron.window = new_window
    held = list(neuron._history)[-new_window:]
    return neuron, sum(held) / len(held)


def _assert_next_step_uses(neuron: SelfReferentialNeuron, rate: float) -> None:
    """Step once at 2.0 and check that ``v`` follows from ``rate``.

    Parameters
    ----------
    neuron : SelfReferentialNeuron
        The neuron to step.
    rate : float
        The rate the step must use.
    """
    v = neuron.v
    candidate = v + (-v + 2.0) / (10.0 * (1.0 + rate / 0.1)) * 1.0
    fired = neuron.step(2.0)
    assert fired == (1 if candidate >= 1.0 else 0)
    assert neuron.v == (0.0 if fired else candidate)


def test_a_shrunk_window_takes_the_rate_over_the_held_steps() -> None:
    """After 50 -> 20 the rate is the spikes of the last 20 held steps over 20, never above 1."""
    neuron, rate = _rate_after_live_change(20)
    assert 0.0 < rate <= 1.0
    _assert_next_step_uses(neuron, rate)


def test_a_grown_window_takes_the_rate_over_the_held_steps() -> None:
    """After 50 -> 100 the rate is the spikes of the 50 held steps over 50 until the history refills."""
    neuron, rate = _rate_after_live_change(100)
    assert len(neuron._history) == 50 and rate > 0.0
    _assert_next_step_uses(neuron, rate)
    assert len(neuron._history) == 51
