# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — independent Poisson statistical reference

"""Re-derive the committed rate, event stream, LFSR period, and seed corpus."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
import pytest

from sc_neurocore.neurons.models.poisson import PoissonNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron, load_schema

_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "src/sc_neurocore/neurons/reference_trace_data/poisson_lfsr16_statistical_v1.json"
)


def _load() -> dict[str, object]:
    return cast(dict[str, object], json.loads(_ARTIFACT.read_text(encoding="utf-8")))


def _independent_events(
    seed: int,
    steps: int,
    probability: float,
) -> tuple[NDArray[np.uint8], int]:
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


def _digest(events: NDArray[np.uint8]) -> str:
    return hashlib.sha256(np.asarray(events, dtype=np.uint8).tobytes()).hexdigest()


def _protocol_probability(protocol: dict[str, object]) -> float:
    parameters = protocol["parameters"]
    assert isinstance(parameters, dict)
    return -math.expm1(-float(parameters["rate_hz"]) * float(parameters["dt_ms"]) / 1000.0)


def _schema_events(
    seed: int,
    steps: int,
    rate_hz: float,
    dt_ms: float,
) -> tuple[NDArray[np.uint8], int]:
    schema = UniversalNeuron.from_schema(
        "poisson",
        parameter_overrides={"rate_hz": rate_hz, "dt_ms": dt_ms},
        rng_seed_override=seed,
    )
    events = np.fromiter((schema.step(I=-1.0) for _ in range(steps)), dtype=np.uint8, count=steps)
    final_state = schema.to_equation_neuron().stochastic_rng_state
    assert final_state is not None
    return events, final_state


def test_reference_provenance_matches_the_paired_source_schemas() -> None:
    artifact = _load()
    schema = load_schema("poisson")
    provenance = artifact["provenance"]
    assert isinstance(provenance, dict)
    assert provenance["citation"] == "doi:10.1017/CBO9781107447615"
    assert schema["metadata"]["doi"] == "10.1017/CBO9781107447615"
    assert schema["threshold"]["detection"] == "poisson"
    assert schema["threshold"]["probability_expression"] == (
        "1.0 - exp(-(((rate_hz if I < 0.0 else I) * dt_ms) / 1000.0))"
    )
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
    probability = _protocol_probability(protocol)
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

    hand = PoissonNeuron(
        rate_hz=float(parameters["rate_hz"]),
        dt_ms=float(parameters["dt_ms"]),
        seed=int(rng["seed"]),
    )
    hand_events, hand_count = hand.simulate(int(protocol["steps"]), backend="python")
    schema_events, schema_state = _schema_events(
        int(rng["seed"]),
        int(protocol["steps"]),
        float(parameters["rate_hz"]),
        float(parameters["dt_ms"]),
    )
    np.testing.assert_array_equal(hand_events, events)
    np.testing.assert_array_equal(schema_events, events)
    assert hand_count == features["spike_count"]
    assert hand.rng_state == schema_state == final_state


def test_seed_corpus_matches_independent_hand_and_schema_streams() -> None:
    artifact = _load()
    protocol = artifact["protocol"]
    corpus = artifact["seed_corpus"]
    assert isinstance(protocol, dict)
    assert isinstance(corpus, list)
    parameters = protocol["parameters"]
    assert isinstance(parameters, dict)
    rate_hz = float(parameters["rate_hz"])
    dt_ms = float(parameters["dt_ms"])
    probability = _protocol_probability(protocol)
    for row in corpus:
        assert isinstance(row, dict)
        seed = int(row["seed"])
        steps = int(row["steps"])
        expected, final_state = _independent_events(seed, steps, probability)
        hand = PoissonNeuron(rate_hz=rate_hz, dt_ms=dt_ms, seed=seed)
        hand_events, hand_count = hand.simulate(steps, backend="python")
        schema_events, schema_state = _schema_events(seed, steps, rate_hz, dt_ms)
        np.testing.assert_array_equal(hand_events, expected)
        np.testing.assert_array_equal(schema_events, expected)
        assert hand_count == row["spike_count"]
        assert hand.rng_state == schema_state == final_state == row["final_rng_state"]
        assert _digest(hand_events) == row["event_sha256"]
