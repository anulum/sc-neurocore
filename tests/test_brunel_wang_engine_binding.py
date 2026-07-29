# SPDX-License-Identifier: AGPL-3.0-or-later
# SC-NeuroCore — installed Brunel-Wang PyO3 boundary contracts

"""Exercise configured construction, full-gate stepping, and reset semantics."""

from __future__ import annotations

import pytest

import sc_neurocore_engine


def test_configured_engine_boundary_and_reset() -> None:
    """Preserve non-default configuration while resetting only dynamic state."""
    neuron = sc_neurocore_engine.BrunelWangNeuron(g_nmda=0.4, dt=0.05)
    assert neuron.step(0.2, 0.1, 0.3, 0.0) in (0, 1)
    neuron.reset()
    assert neuron.get_state() == (-70.0, 0.0)


def test_engine_failure_is_atomic() -> None:
    """Reject invalid aggregate input without mutating dynamic state."""
    neuron = sc_neurocore_engine.BrunelWangNeuron(v=-63.0)
    before = neuron.get_state()
    with pytest.raises(ValueError):
        neuron.step(float("nan"), 0.0, 0.0, 0.0)
    assert neuron.get_state() == before
