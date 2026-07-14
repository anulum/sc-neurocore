# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — independent EscapeRate statistical reference

"""Re-derive the committed rate, event stream, LFSR period, and seed corpus."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.neurons.models.escape_rate import EscapeRateNeuron
from sc_neurocore.neurons.universal_dsl import load_schema

_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "src/sc_neurocore/neurons/reference_trace_data/escape_rate_lfsr16_statistical_v1.json"
)


def _load() -> dict[str, object]:
    return json.loads(_ARTIFACT.read_text(encoding="utf-8"))


def _independent_events(seed: int, steps: int, probability: float) -> tuple[np.ndarray, int]:
    """Evaluate the documented polynomial without importing production RNG code."""
    threshold = (
        0
        if probability <= 0.0
        else 65_536
        if probability >= 1.0
        else math.floor(probability * 65_535.0) + 1
    )
    state = seed
    events = np.empty(steps, dtype=np.uint8)
    for index in range(steps):
        for _ in range(8):
            feedback = ((state >> 0) ^ (state >> 2) ^ (state >> 3) ^ (state >> 5)) & 1
            state = ((state >> 1) | (feedback << 15)) & 0xFFFF
        events[index] = state < threshold
    return events, state


def _digest(events: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(events, dtype=np.uint8).tobytes()).hexdigest()


def test_reference_provenance_matches_the_paired_source_schemas() -> None:
    artifact = _load()
    schema = load_schema("escape_rate")
    provenance = artifact["provenance"]
    assert isinstance(provenance, dict)
    assert provenance["citation"] == "doi:10.1162/089976600300015899"
    assert schema["metadata"]["doi"] == "10.1162/089976600300015899"
    assert schema["threshold"]["detection"] == "escape_rate"
    assert schema["threshold"]["rng_seed"] == 0xACE1


def test_full_period_features_match_independent_equations() -> None:
    artifact = _load()
    protocol = artifact["protocol"]
    features = artifact["expected_features"]
    assert isinstance(protocol, dict)
    assert isinstance(features, dict)
    parameters = protocol["parameters"]
    rng = protocol["rng"]
    assert isinstance(parameters, dict)
    assert isinstance(rng, dict)
    hazard = float(parameters["rho_0"]) * float(protocol["dt"])
    probability = -math.expm1(-hazard)
    events, final_state = _independent_events(int(rng["seed"]), int(protocol["steps"]), probability)
    indices = np.flatnonzero(events)
    intervals = np.diff(indices)

    assert probability == pytest.approx(features["continuous_probability"], abs=0.0)
    assert int(events.sum()) == features["spike_count"] == 14_496
    assert float(events.mean()) == pytest.approx(features["realised_probability"], abs=0.0)
    assert int(indices[0]) == features["first_spike_step"]
    assert int(indices[-1]) == features["last_spike_step"]
    assert final_state == features["final_rng_state"] == rng["seed"]
    assert _digest(events) == features["event_sha256"]
    assert float(intervals.mean()) == pytest.approx(features["mean_isi_steps"], abs=1.0e-15)
    assert float(intervals.std()) == pytest.approx(features["std_isi_steps"], abs=1.0e-15)
    assert float(intervals.std() / intervals.mean()) == pytest.approx(
        features["cv_isi"], abs=1.0e-15
    )


def test_seed_corpus_matches_independent_and_production_streams() -> None:
    artifact = _load()
    protocol = artifact["protocol"]
    corpus = artifact["seed_corpus"]
    assert isinstance(protocol, dict)
    assert isinstance(corpus, list)
    parameters = protocol["parameters"]
    assert isinstance(parameters, dict)
    probability = -math.expm1(-float(parameters["rho_0"]) * float(protocol["dt"]))
    for row in corpus:
        assert isinstance(row, dict)
        seed = int(row["seed"])
        steps = int(row["steps"])
        expected, final_state = _independent_events(seed, steps, probability)
        production = EscapeRateNeuron(
            v=-50.0,
            v_rest=-50.0,
            v_reset=-50.0,
            v_threshold=-50.0,
            rho_0=0.25,
            delta_u=1.0,
            dt=1.0,
            seed=seed,
        )
        actual = np.fromiter(
            (production.step(0.0) for _ in range(steps)),
            dtype=np.uint8,
            count=steps,
        )
        np.testing.assert_array_equal(actual, expected)
        assert int(actual.sum()) == row["spike_count"]
        assert production.rng_state == final_state == row["final_rng_state"]
        assert _digest(actual) == row["event_sha256"]
