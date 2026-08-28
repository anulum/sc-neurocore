# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Independent Courbage source-profile reference

"""Independent Figure-4 equation and receipt evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.neurons.reference_traces import (
    load_reference_trace_spec,
    validate_reference_trace_spec,
)

ROOT = Path(__file__).resolve().parents[1]


def _independent_orbit(steps: int) -> tuple[list[float], list[float], list[int]]:
    """Iterate source equations 3–5 with the complete Figure-4 profile."""
    m0 = 0.4
    m1 = 0.65
    a = 0.2
    d = 0.3
    j = 0.13
    beta = 0.25
    epsilon = 0.002
    threshold = d
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
        x_next = x + field - y - beta * (1.0 if x >= d else 0.0)
        y_next = y + epsilon * (x - j)
        events.append(int(x_next >= threshold and x < threshold))
        x, y = x_next, y_next
        x_values.append(x)
        y_values.append(y)
    return x_values, y_values, events


def _features(steps: int) -> dict[str, float]:
    x_values, y_values, events = _independent_orbit(steps)
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


def test_figure_four_profile_satisfies_source_region() -> None:
    """Figure-4 defaults must satisfy equations 6 and the stated m0 bound."""
    m0, m1, a, d, j = 0.4, 0.65, 0.2, 0.3, 0.13
    j_min = a * m1 / (m0 + m1)
    j_max = (m0 + a * m1) / (m0 + m1)
    assert 0.0 < j < d
    assert j_min < d < j_max
    assert m0 < 1.0


def test_features_match_independent_source_profile() -> None:
    """Committed features must match the independently transcribed source map."""
    spec = load_reference_trace_spec("courage_nekorkin_map_autonomous_doi")
    expected = _features(spec.protocol.steps)
    assert spec.schema_name == "courage_nekorkin_map"
    assert spec.provenance.kind == "source_equation_reference"
    assert spec.provenance.citation == "doi:10.1063/1.2795435"
    assert expected["spike_count"] == 63.0
    assert expected["first_spike_step"] == 618.0
    assert set(expected) == set(spec.expected_features)
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)


def test_source_receipt_digest_is_independently_reproducible() -> None:
    """The source receipt must bind the complete Figure-4 x trajectory."""
    receipt = json.loads(
        (
            ROOT / "src/sc_neurocore/neurons/reference_receipts/courbage_nekorkin_vdovin_2007.json"
        ).read_text()
    )
    x_values, y_values, events = _independent_orbit(receipt["drive"]["steps"])
    digest = hashlib.sha256(np.asarray(x_values, dtype="<f8").tobytes()).hexdigest()
    assert sum(events) == receipt["oracle"]["events"]
    assert x_values[-1] == receipt["oracle"]["x_final"]
    assert y_values[-1] == receipt["oracle"]["y_final"]
    assert digest == receipt["oracle"]["trace_sha256"]


def test_committed_trace_validates_through_schema_runner() -> None:
    """The production schema runner must reproduce the independent contract."""
    report = validate_reference_trace_spec(
        load_reference_trace_spec("courage_nekorkin_map_autonomous_doi")
    )
    assert report.passed
    assert report.mismatches == ()
