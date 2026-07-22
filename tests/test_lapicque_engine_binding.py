# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Lapicque engine-binding contracts

"""Installed-extension contracts for the canonical Lapicque binding."""

from __future__ import annotations

import importlib
import math

import pytest

import sc_neurocore_engine as engine

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def test_exported_class_signature_and_top_level_identity_are_stable() -> None:
    lapicque = extension.LapicqueNeuron

    assert lapicque.__name__ == "LapicqueNeuron"
    assert lapicque.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert lapicque.__text_signature__ == ("(tau=20.0, resistance=1.0, threshold=1.0, dt=1.0)")
    assert engine.LapicqueNeuron is lapicque


def test_default_step_matches_independent_exact_rc_flow() -> None:
    neuron = extension.LapicqueNeuron()
    current = 0.25
    v_inf = current
    expected_v = v_inf + (0.0 - v_inf) * math.exp(-1.0 / 20.0)

    assert neuron.step(current) == 0
    assert neuron.get_state() == pytest.approx({"v": expected_v}, rel=0.0, abs=1e-15)


def test_threshold_crossing_and_reset_restore_zero_state() -> None:
    neuron = extension.LapicqueNeuron(tau=2.0, resistance=2.0, threshold=1.0, dt=1.0)

    spikes = [neuron.step(1.0) for _ in range(8)]
    assert 1 in spikes

    neuron.step(0.5)
    neuron.reset()
    assert neuron.get_state() == pytest.approx({"v": 0.0})
