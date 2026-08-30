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

import numpy as np
import pytest

from sc_neurocore.neurons.models.lapicque import LapicqueNeuron
from tests.engine_requirement import require_engine

require_engine()
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


def test_complete_binding_is_exported_through_the_public_bridge() -> None:
    assert engine.lapicque_simulate_complete is extension.lapicque_simulate_complete
    assert "lapicque_simulate_complete" in engine.__all__


def test_complete_binding_transports_source_profile_and_event_latch() -> None:
    voltage, events, final_v, excited = extension.lapicque_simulate_complete(
        0.0,
        0.0,
        0.0,
        1.0,
        20.0,
        1.0,
        0.01,
        1.1,
        10.0,
        1.0,
        False,
        True,
        2_000,
        22.0,
    )
    assert voltage.shape == events.shape == (2_000,)
    assert events.dtype == np.uint8
    assert np.flatnonzero(events).tolist() == [69]
    assert final_v == voltage[-1]
    assert excited is True


def test_complete_binding_rejection_is_failure_atomic() -> None:
    with pytest.raises(FloatingPointError, match="Lapicque batch rejected"):
        extension.lapicque_simulate_complete(
            0.25,
            0.0,
            0.0,
            1.0,
            20.0,
            1.0,
            0.01,
            1.1,
            10.0,
            1.0,
            False,
            True,
            2,
            math.nan,
        )


def test_network_runner_canonical_spelling_selects_the_source_profile() -> None:
    python = LapicqueNeuron.lapicque_1907()
    expected_event = python.step(22.0)
    runner = extension.NetworkRunner()
    population = runner.add_population("LapicqueNeuron", 1)
    result = runner.step_population(population, np.array([22.0], dtype=np.float64))
    assert result["spikes"].tolist() == [expected_event]
    assert result["voltages"].tolist() == pytest.approx([python.v], abs=1.0e-15)


@pytest.mark.parametrize("alias", ("Lapicque", "SCLapicqueLIF", "SCLapicqueLIFNeuron"))
def test_network_runner_aliases_preserve_the_sc_profile(alias: str) -> None:
    python = LapicqueNeuron.sc_lif_compatibility()
    expected_event = python.step(5.0)
    runner = extension.NetworkRunner()
    population = runner.add_population(alias, 1)
    result = runner.step_population(population, np.array([5.0], dtype=np.float64))
    assert result["spikes"].tolist() == [expected_event]
    assert result["voltages"].tolist() == pytest.approx([python.v], abs=1.0e-15)
