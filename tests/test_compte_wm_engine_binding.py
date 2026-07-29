# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Compte PyO3 boundary contracts

"""Focused constructor, state, error, and source-default checks for PyO3."""

from __future__ import annotations

import pytest

from sc_neurocore_engine import CompteWMNeuron


def test_binding_exposes_complete_source_state() -> None:
    state = CompteWMNeuron()
    assert state.get_state() == {
        "v": -70.0,
        "s_ampa": 0.0,
        "s_nmda": 0.0,
        "x_nmda": 0.0,
        "s_gaba": 0.0,
        "ref_remaining": 0.0,
    }
    assert state.step(0.0, True, False, False) == 0
    dynamic = state.get_state()
    assert dynamic["s_ampa"] == 0.0
    assert dynamic["s_nmda"] > 0.0
    assert dynamic["x_nmda"] > 0.0
    assert dynamic["s_gaba"] == 0.0


def test_binding_constructor_is_configurable_and_fail_closed() -> None:
    state = CompteWMNeuron(v=-50.01, dt=0.01, tau_gaba=9.0)
    assert state.step(1.0) == 1
    assert state.get_state()["ref_remaining"] == 2.0
    with pytest.raises(ValueError):
        CompteWMNeuron(dt=0.0)
    before = state.get_state()
    with pytest.raises(ValueError):
        state.step(float("nan"))
    assert state.get_state() == before


def test_binding_reset_preserves_configuration_behavior() -> None:
    state = CompteWMNeuron(dt=0.01)
    state.step(1.0, True, True, True)
    state.reset()
    assert tuple(state.get_state().values()) == (-70.0, 0.0, 0.0, 0.0, 0.0, 0.0)
