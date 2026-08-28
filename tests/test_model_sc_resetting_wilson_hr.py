# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — historical resetting Wilson-HR identity contracts

"""Preservation tests for the pre-split SC resetting Wilson-HR recurrence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.neurons.models.sc_resetting_wilson_hr import SCResettingWilsonHRNeuron
from sc_neurocore.neurons.models.wilson_hr import WilsonHRNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron


def test_historical_defaults_and_one_step_are_preserved() -> None:
    neuron = SCResettingWilsonHRNeuron()
    assert (neuron.v, neuron.r, neuron.tau_r, neuron.v_peak, neuron.dt) == (
        -0.7,
        0.1,
        1.9,
        0.4,
        0.05,
    )
    assert neuron.step(2.0) == 0
    assert neuron.v == -0.5988676025214146
    assert neuron.r == 0.10134793845659071


def test_historical_trace_and_event_anchor_are_preserved() -> None:
    neuron = SCResettingWilsonHRNeuron()
    trace, spikes = neuron.simulate(1_000, 2.0)
    assert spikes == 1
    assert trace.shape == (1_000,)
    assert neuron.v == -0.7238362204839788
    assert neuron.r == 0.5188711685571015


def test_sc_identity_is_distinct_from_source_wilson_flow() -> None:
    sc = SCResettingWilsonHRNeuron()
    source = WilsonHRNeuron()
    assert sc.step(0.1) == source.step(0.1) == 0
    assert sc.v != source.v
    assert sc.r != source.r


def test_invalid_batch_is_failure_atomic() -> None:
    neuron = SCResettingWilsonHRNeuron()
    before = (neuron.v, neuron.r)
    with pytest.raises(FloatingPointError, match="candidate|derivative"):
        neuron.simulate(2, 1.0e308)
    assert (neuron.v, neuron.r) == before


def test_paired_schemas_match_historical_project_recurrence() -> None:
    schema_dir = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
    hand = SCResettingWilsonHRNeuron()
    toml_model = UniversalNeuron.from_schema(schema_dir / "sc_resetting_wilson_hr.toml")
    json_model = UniversalNeuron.from_schema(schema_dir / "sc_resetting_wilson_hr.json")
    for current in np.resize(np.array([0.0, 2.0, 10.0, 5.0]), 400):
        event = hand.step(float(current))
        assert int(bool(toml_model.step(I=float(current)))) == event
        assert int(bool(json_model.step(I=float(current)))) == event
        assert toml_model.state == json_model.state == {"v": hand.v, "r": hand.r}
