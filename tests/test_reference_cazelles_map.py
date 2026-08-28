# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Independent Cazelles source-map reference

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


def _independent_source_orbit(steps: int) -> tuple[list[float], list[int]]:
    x = 0.1
    values: list[float] = []
    events: list[int] = []
    for _ in range(steps):
        previous = x
        if x < 0.4:
            x = 1.05 * x
        elif x < 0.6:
            x = 1.5 - 1.25 * x
        elif x < 0.7:
            x = -0.9 + 1.5 * x
        else:
            x = 1.4 - x
        events.append(int(previous >= 0.4 and x < 0.4))
        values.append(x)
    return values, events


def test_committed_features_match_independent_primary_equation() -> None:
    spec = load_reference_trace_spec("cazelles_map_bursting_doi")
    values, events = _independent_source_orbit(spec.protocol.steps)
    expected = {
        "spike_count": float(math.fsum(events)),
        "first_spike_step": float(next(index for index, event in enumerate(events, 1) if event)),
        "final.x": values[-1],
        "min.x": min(values),
        "max.x": max(values),
        "mean.x": math.fsum(values) / len(values),
    }
    assert spec.schema_name == "cazelles_map"
    assert spec.provenance.kind == "source_equation_reference"
    assert spec.provenance.citation == "doi:10.1209/epl/i2001-00548-y"
    assert expected["spike_count"] == 7.0
    assert expected["first_spike_step"] == 56.0
    assert set(expected) == set(spec.expected_features)
    for feature, value in expected.items():
        assert spec.expected_features[feature] == pytest.approx(value, abs=1.0e-12)


def test_source_receipt_trace_digest_is_independently_reproducible() -> None:
    receipt = json.loads(
        (ROOT / "src/sc_neurocore/neurons/reference_receipts/cazelles_2001.json").read_text()
    )
    values, events = _independent_source_orbit(receipt["drive"]["steps"])
    digest = hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()
    assert sum(events) == receipt["oracle"]["events"]
    assert digest == receipt["oracle"]["trace_sha256"]


def test_committed_reference_validates_through_schema_runner() -> None:
    report = validate_reference_trace_spec(load_reference_trace_spec("cazelles_map_bursting_doi"))
    assert report.passed
    assert report.mismatches == ()
