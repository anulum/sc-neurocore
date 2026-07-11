# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Independent Courbage-Nekorkin map reference trace

"""Independent feature derivation for the committed Courbage-Nekorkin trace."""

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.reference_traces import (
    load_reference_trace_spec,
    validate_reference_trace_spec,
)


def _independent_features(*, current: float, steps: int) -> dict[str, float]:
    """Iterate Courbage et al. (2007), equations 3-5, without model code."""
    m0 = 0.0864
    m1 = 0.65
    a = 0.2
    d = 0.235
    j = 0.2
    beta = 0.085
    epsilon = 0.02
    threshold = 0.235
    j_min = a * m1 / (m0 + m1)
    j_max = (m0 + a * m1) / (m0 + m1)
    x = 0.0
    y = 0.0
    x_values: list[float] = []
    y_values: list[float] = []
    events: list[int] = []

    for _step in range(steps):
        if x <= j_min:
            field = -m0 * x
        elif x < j_max:
            field = m1 * (x - a)
        else:
            field = -m0 * (x - 1.0)
        x_next = x + field - y - beta * (1.0 if x >= d else 0.0) + current
        y_next = y + epsilon * (x - j)
        events.append(1 if x_next >= threshold and x < threshold else 0)
        x, y = x_next, y_next
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


def test_default_parameters_satisfy_published_map_region() -> None:
    """Defaults must satisfy Courbage et al. equations 6, 9, and 12."""
    m0 = 0.0864
    m1 = 0.65
    a = 0.2
    d = 0.235
    j = 0.2
    beta = 0.085
    j_min = a * m1 / (m0 + m1)
    j_max = (m0 + a * m1) / (m0 + m1)
    field_min = -m0 * j_min
    field_max = -m0 * (j_max - 1.0)
    beta_min = field_max - field_min
    q = 1.0 + m1
    beta_max = min(q * (j_max - d), q * (d - j_min))

    assert 0.0 < j < d
    assert j_min < d < j_max
    assert m0 < 1.0
    assert beta_min < beta < beta_max


def test_features_match_independent_map_iteration() -> None:
    """Committed features must match an independent autonomous recurrence."""
    spec = load_reference_trace_spec("courage_nekorkin_map_autonomous_doi")
    expected = _independent_features(
        current=spec.protocol.inputs["I"],
        steps=spec.protocol.steps,
    )

    assert spec.schema_name == "courage_nekorkin_map"
    assert spec.provenance.kind == "map_iteration_reference"
    assert spec.provenance.citation == "doi:10.1063/1.2795435"
    assert expected["spike_count"] == 4.0
    assert set(expected) == set(spec.expected_features)
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)


def test_committed_trace_validates_through_schema_runner() -> None:
    """The committed feature contract must pass the production schema runner."""
    spec = load_reference_trace_spec("courage_nekorkin_map_autonomous_doi")

    report = validate_reference_trace_spec(spec)

    assert report.passed
    assert report.mismatches == ()
