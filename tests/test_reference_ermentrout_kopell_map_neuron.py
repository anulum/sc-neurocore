# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Independent Ermentrout-Kopell theta-Euler reference

"""Independent feature derivation for the maintained theta-Euler recurrence."""

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.reference_traces import (
    load_reference_trace_spec,
    validate_reference_trace_spec,
)


def _independent_features(*, current: float, steps: int) -> dict[str, float]:
    """Advance the sourced theta flow with the maintained Euler/wrap convention."""
    dt = 0.1
    gain = 1.0
    threshold = math.pi
    two_pi = 2.0 * math.pi
    theta = 0.0
    values: list[float] = []
    events: list[int] = []

    for _step in range(steps):
        previous = theta
        derivative = (1.0 - math.cos(previous)) + ((1.0 + math.cos(previous)) * gain * current)
        candidate = previous + dt * derivative
        events.append(1 if previous < threshold <= candidate else 0)
        theta = candidate % two_pi
        values.append(theta)

    return {
        "spike_count": float(math.fsum(events)),
        "first_spike_step": float(
            next((index for index, event in enumerate(events, start=1) if event), -1)
        ),
        "final.theta": values[-1],
        "min.theta": min(values),
        "max.theta": max(values),
        "mean.theta": math.fsum(values) / len(values),
    }


def test_features_match_independent_theta_euler_iteration() -> None:
    """Committed features must match an independent 1986-theta/Euler derivation."""
    spec = load_reference_trace_spec("ermentrout_kopell_theta_euler_doi")
    expected = _independent_features(
        current=spec.protocol.inputs["I"],
        steps=spec.protocol.steps,
    )

    assert spec.schema_name == "ermentrout_kopell_map_neuron"
    assert spec.provenance.kind == "independent_euler_reference"
    assert spec.provenance.citation == "doi:10.1137/0146017"
    assert expected["spike_count"] == 45.0
    assert expected["first_spike_step"] == 23.0
    assert set(expected) == set(spec.expected_features)
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)


def test_committed_trace_validates_through_schema_runner() -> None:
    """The independent feature contract must pass through the production runner."""
    spec = load_reference_trace_spec("ermentrout_kopell_theta_euler_doi")

    report = validate_reference_trace_spec(spec)

    assert report.passed
    assert report.mismatches == ()
