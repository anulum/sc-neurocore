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

import sc_neurocore_engine as engine

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def test_exported_class_signature_and_top_level_identity_are_stable() -> None:
    adex = extension.AdExNeuron

    assert adex.__name__ == "AdExNeuron"
    assert adex.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert adex.__text_signature__ == "()"
    assert engine.AdExNeuron is adex


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
