# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ExpIF production binding and NetworkRunner contracts

from __future__ import annotations

import importlib

import numpy as np
import pytest

from sc_neurocore.neurons.models.expif import ExpIFNeuron
from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def test_complete_binding_is_exported_through_the_public_bridge() -> None:
    assert engine.expif_simulate_complete is extension.expif_simulate_complete
    assert "expif_simulate_complete" in engine.__all__


def test_complete_binding_transports_source_profile_and_all_rows() -> None:
    result = extension.expif_simulate_complete(
        -65.0,
        -65.0,
        -68.0,
        -30.0,
        -59.9,
        3.48,
        10.0,
        0.01,
        1.7,
        0.0,
        True,
        4_000,
        20.0,
    )
    voltage, refractory, events, final_v, final_refractory = result

    assert voltage.shape == refractory.shape == events.shape == (4_000,)
    assert events.dtype == np.uint8
    assert int(events.sum()) > 0
    assert (final_v, final_refractory) == pytest.approx(
        (voltage[-1], refractory[-1]), rel=0.0, abs=0.0
    )


def test_complete_binding_rejects_a_false_source_profile() -> None:
    with pytest.raises(FloatingPointError, match="ExpIF batch rejected"):
        extension.expif_simulate_complete(
            -65.0,
            -65.0,
            -68.0,
            -30.0,
            -59.9,
            3.48,
            10.0,
            0.02,
            1.7,
            0.0,
            True,
            1,
            20.0,
        )


def test_network_runner_canonical_spelling_selects_the_source_profile() -> None:
    python = ExpIFNeuron.fourcaud_trocme_2003()
    expected_event = python.step(20.0)

    runner = extension.NetworkRunner()
    population = runner.add_population("ExpIFNeuron", 1)
    result = runner.step_population(population, np.array([20.0], dtype=np.float64))

    assert result["spikes"].tolist() == [expected_event]
    assert result["voltages"].tolist() == pytest.approx([python.v], rel=0.0, abs=1.0e-12)


@pytest.mark.parametrize("alias", ("ExpIF", "ExpIfNeuron"))
def test_network_runner_legacy_aliases_preserve_sc_rk4_profile(alias: str) -> None:
    python = ExpIFNeuron.sc_rk4_compatibility()
    expected_event = python.step(20.0)

    runner = extension.NetworkRunner()
    population = runner.add_population(alias, 1)
    result = runner.step_population(population, np.array([20.0], dtype=np.float64))

    assert result["spikes"].tolist() == [expected_event]
    assert result["voltages"].tolist() == pytest.approx([python.v], rel=0.0, abs=1.0e-12)
