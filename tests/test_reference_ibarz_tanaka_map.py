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
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.neurons.reference_traces import (
    load_reference_trace_spec,
    validate_reference_trace_spec,
)

ROOT = Path(__file__).resolve().parents[1]


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
            event = 0
        elif v <= 0.0:
            v_next = alpha * v + (v + 1.0) * (v + 1.0) + current + u
            event = 0
        elif v < upper:
            v_next = upper
            event = 0
        else:
            v_next = -1.0
            event = 1
        u_next = u - mu * (v + 1.0 - sigma)
        events.append(event)
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


def test_complete_state_and_event_receipt_is_independently_reproducible() -> None:
    """The two-source receipt binds every state and reset decision."""
    receipt = json.loads(
        (
            ROOT / "src/sc_neurocore/neurons/reference_receipts/ibarz_tanaka_shilnikov_rulkov.json"
        ).read_text(encoding="utf-8")
    )
    profile = receipt["analysis_profile"]
    alpha = float(profile["parameters"]["alpha"])
    mu = float(profile["parameters"]["mu"])
    sigma = float(profile["parameters"]["sigma"])
    current = float(receipt["drive"]["current"])
    v = float(profile["initial_state"]["v"])
    u = float(profile["initial_state"]["u"])
    states: list[tuple[float, float]] = []
    events: list[int] = []

    for _ in range(int(receipt["drive"]["steps"])):
        lower = -1.0 - alpha / 2.0
        upper = 1.0 + current + u
        if v < lower:
            v_next = -(alpha * alpha) / 4.0 - alpha + current + u
            event = 0
        elif v <= 0.0:
            v_next = alpha * v + (v + 1.0) * (v + 1.0) + current + u
            event = 0
        elif v < upper:
            v_next = upper
            event = 0
        else:
            v_next = -1.0
            event = 1
        u_next = u - mu * (v + 1.0 - sigma)
        events.append(event)
        v, u = v_next, u_next
        states.append((v, u))

    state_digest = hashlib.sha256(np.asarray(states, dtype="<f8").tobytes(order="C")).hexdigest()
    event_digest = hashlib.sha256(np.asarray(events, dtype=np.uint8).tobytes()).hexdigest()
    oracle = receipt["oracle"]
    assert receipt["equation_origin"]["citation"] == "doi:10.1016/j.physleta.2004.05.062"
    assert profile["citation"] == "doi:10.1103/PhysRevE.75.041902"
    assert sum(events) == oracle["events"]
    assert events.index(1) == oracle["first_event_index"]
    assert {"v": v, "u": u} == oracle["final_state"]
    assert state_digest == oracle["state_trace_sha256"]
    assert event_digest == oracle["event_trace_sha256"]
