# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — AdEx engine-binding contracts

"""Installed-extension contracts for the canonical AdEx binding."""

from __future__ import annotations

import importlib
import math

import pytest

from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def test_exported_class_signature_and_top_level_identity_are_stable() -> None:
    adex = extension.AdExNeuron

    assert adex.__name__ == "AdExNeuron"
    assert adex.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert adex.__text_signature__ == "()"
    assert engine.AdExNeuron is adex
    assert engine.adex_simulate_complete is extension.adex_simulate_complete


def test_default_step_matches_independent_brette_gerstner_update() -> None:
    neuron = extension.AdExNeuron()
    initial_v = -65.0
    initial_w = 0.0
    current = 25.0
    exp_term = 2.0 * math.exp((initial_v - -55.0) / 2.0)
    expected_v = (
        initial_v
        + ((-(initial_v - -65.0) + exp_term) / 20.0 + (-initial_w + current) / 200.0) * 0.1
    )
    expected_w = initial_w + (0.5 * (initial_v - -65.0) - initial_w) / 100.0 * 0.1

    assert neuron.step(current) == 0
    assert neuron.get_state() == pytest.approx(
        {"v": expected_v, "w": expected_w}, rel=0.0, abs=1e-12
    )


def test_reset_restores_default_state_after_sustained_drive() -> None:
    neuron = extension.AdExNeuron()
    for _ in range(64):
        neuron.step(500.0)

    neuron.reset()

    assert neuron.get_state() == pytest.approx({"v": -65.0, "w": 0.0})


def test_complete_binding_transports_nondefault_state_and_parameters() -> None:
    """Exercise the real checked Rust batch rather than a factory-step loop."""
    result = extension.adex_simulate_complete(
        -60.0,
        3.0,
        -64.0,
        -69.0,
        -49.0,
        -54.0,
        2.5,
        18.0,
        120.0,
        0.7,
        8.0,
        180.0,
        0.2,
        250,
        410.0,
    )
    v_trace, w_trace, event_trace, final_v, final_w = result

    assert v_trace.shape == w_trace.shape == event_trace.shape == (250,)
    assert int(event_trace.sum()) == 5
    assert (final_v, final_w) == pytest.approx((v_trace[-1], w_trace[-1]))


def test_complete_binding_rejects_nonfinite_batch_without_packet() -> None:
    """Surface checked-Rust arithmetic failure as a Python exception."""
    with pytest.raises(FloatingPointError, match="non-finite"):
        extension.adex_simulate_complete(
            -65.0,
            0.0,
            -65.0,
            -68.0,
            -50.0,
            -55.0,
            2.0,
            20.0,
            100.0,
            0.5,
            7.0,
            200.0,
            1.0e308,
            2,
            1.0e308,
        )
