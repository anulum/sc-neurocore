# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Independent Cazelles map reference trace

"""Independent feature derivation for the committed Cazelles map trace."""

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.reference_traces import (
    load_reference_trace_spec,
    validate_reference_trace_spec,
)


def _independent_features(*, current: float, steps: int) -> dict[str, float]:
    """Iterate the simultaneous clipped map without calling model code."""
    a = 3.8
    epsilon = 0.01
    sigma = 0.5
    threshold = 0.9
    x = 0.1
    y = 0.0
    x_values: list[float] = []
    y_values: list[float] = []
    events: list[int] = []

    for _step in range(steps):
        x_next = min(2.0, max(-2.0, a * x * (1.0 - x) - y + current))
        y_next = y + epsilon * (x - sigma)
        x, y = x_next, y_next
        events.append(1 if x >= threshold else 0)
        x_values.append(x)
        y_values.append(y)

    return {
        "spike_count": float(math.fsum(events)),
        "first_spike_step": float(
            next((index for index, event in enumerate(events, start=1) if event), -1)
        ),
        "final.x": x_values[-1],
        "min.x": min(x_values),
        "max.x": max(x_values),
        "mean.x": math.fsum(x_values) / len(x_values),
        "final.y": y_values[-1],
        "min.y": min(y_values),
        "max.y": max(y_values),
        "mean.y": math.fsum(y_values) / len(y_values),
    }


def test_features_match_independent_map_iteration() -> None:
    """Committed features must match an independent simultaneous recurrence."""
    spec = load_reference_trace_spec("cazelles_map_bursting_doi")
    expected = _independent_features(
        current=spec.protocol.inputs["I"],
        steps=spec.protocol.steps,
    )

    assert spec.schema_name == "cazelles_map"
    assert spec.provenance.kind == "map_iteration_reference"
    assert spec.provenance.citation == "doi:10.1209/epl/i2001-00548-y"
    assert expected["spike_count"] == 2.0
    assert set(expected) == set(spec.expected_features)
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)


def test_committed_trace_validates_through_schema_runner() -> None:
    """The committed feature contract must pass the production schema runner."""
    spec = load_reference_trace_spec("cazelles_map_bursting_doi")

    report = validate_reference_trace_spec(spec)

    assert report.passed
    assert report.mismatches == ()
