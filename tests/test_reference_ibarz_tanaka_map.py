# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Independent Ibarz-Tanaka 2007 reference trace

"""Independent equation-level derivation of the committed DOI trace."""

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.reference_traces import (
    load_reference_trace_spec,
    validate_reference_trace_spec,
)


def _independent_features(*, current: float, steps: int) -> dict[str, float]:
    """Iterate Eqs. 2-3 without importing the production model."""
    alpha = 1.0
    mu = 0.001
    sigma = 0.1
    v = -1.0
    u = -0.1
    values_v: list[float] = []
    values_u: list[float] = []
    events: list[int] = []

    for _step in range(steps):
        lower = -1.0 - alpha / 2.0
        upper = 1.0 + current + u
        if v < lower:
            v_next = -(alpha * alpha) / 4.0 - alpha + current + u
        elif v <= 0.0:
            v_next = alpha * v + (v + 1.0) * (v + 1.0) + current + u
        elif v < upper:
            v_next = upper
        else:
            v_next = -1.0
        u_next = u - mu * (v + 1.0 - sigma)
        events.append(int(v >= upper))
        v, u = v_next, u_next
        values_v.append(v)
        values_u.append(u)

    first_event = next((index for index, event in enumerate(events, start=1) if event), -1)
    return {
        "spike_count": float(math.fsum(events)),
        "first_spike_step": float(first_event),
        "final.v": values_v[-1],
        "min.v": min(values_v),
        "max.v": max(values_v),
        "mean.v": math.fsum(values_v) / len(values_v),
        "final.u": values_u[-1],
        "min.u": min(values_u),
        "max.u": max(values_u),
        "mean.u": math.fsum(values_u) / len(values_u),
    }


def test_features_match_independent_eq_2_3_iteration() -> None:
    """Committed features must match a fresh primary-equation derivation."""
    spec = load_reference_trace_spec("ibarz_tanaka_map_2007_doi")
    expected = _independent_features(
        current=spec.protocol.inputs["I"],
        steps=spec.protocol.steps,
    )

    assert spec.schema_name == "ibarz_tanaka_map"
    assert spec.provenance.kind == "map_iteration_reference"
    assert spec.provenance.citation == "doi:10.1103/PhysRevE.75.041902"
    assert expected["spike_count"] == 9.0
    assert expected["first_spike_step"] == 395.0
    assert set(expected) == set(spec.expected_features)
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)


def test_committed_trace_validates_through_schema_runner() -> None:
    """The production schema runner must reproduce the feature contract."""
    spec = load_reference_trace_spec("ibarz_tanaka_map_2007_doi")
    report = validate_reference_trace_spec(spec)
    assert report.passed
    assert report.mismatches == ()
