# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Independent retained rational-recovery reference

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
    x = 0.0
    y = 0.0
    alpha = 3.0
    beta = 0.001
    offset = 0.1
    threshold = 1.0
    bound = 1_000_000.0
    x_values: list[float] = []
    y_values: list[float] = []
    events: list[int] = []
    for _step in range(steps):
        field = alpha * x if x < 0.0 else alpha * x / (1.0 + alpha * x)
        x_new = min(bound, max(-bound, field + y + offset))
        y_new = min(bound, max(-bound, y - beta * (x + 1.0)))
        events.append(int(x_new >= threshold and x < threshold))
        x, y = x_new, y_new
        x_values.append(x)
        y_values.append(y)
    return x_values, y_values, events


def test_project_features_and_receipt_are_reproducible() -> None:
    spec = load_reference_trace_spec("sc_clipped_rational_recovery_map_project")
    x_values, y_values, events = _independent_orbit(spec.protocol.steps)
    expected = {
        "spike_count": float(math.fsum(events)),
        "first_spike_step": -1.0,
        "final.x": x_values[-1],
        "min.x": min(x_values),
        "max.x": max(x_values),
        "mean.x": math.fsum(x_values) / len(x_values),
        "final.y": y_values[-1],
        "min.y": min(y_values),
        "max.y": max(y_values),
        "mean.y": math.fsum(y_values) / len(y_values),
    }
    assert spec.provenance.kind == "project_regression"
    for feature, value in expected.items():
        assert spec.expected_features[feature] == pytest.approx(value, abs=1e-8)

    receipt = json.loads(
        (
            ROOT
            / "src/sc_neurocore/neurons/reference_receipts/sc_clipped_rational_recovery_map_project.json"
        ).read_text()
    )
    digest = hashlib.sha256(np.asarray(x_values, dtype="<f8").tobytes()).hexdigest()
    assert sum(events) == receipt["oracle"]["events"]
    assert x_values[-1] == receipt["oracle"]["x_final"]
    assert y_values[-1] == receipt["oracle"]["y_final"]
    assert digest == receipt["oracle"]["trace_sha256"]


def test_project_reference_validates_through_schema_runner() -> None:
    report = validate_reference_trace_spec(
        load_reference_trace_spec("sc_clipped_rational_recovery_map_project")
    )
    assert report.passed
    assert report.mismatches == ()
