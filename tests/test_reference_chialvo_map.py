# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Independent Chialvo map reference trace

"""Independent feature derivation for the committed Chialvo DOI trace."""

from __future__ import annotations

import math
import hashlib
import json
import struct
from pathlib import Path

import pytest

from sc_neurocore.neurons.reference_traces import (
    load_reference_trace_spec,
    validate_reference_trace_spec,
)


def _independent_features(*, current: float, steps: int) -> dict[str, float]:
    """Iterate Chialvo's simultaneous source recurrence without model code."""
    a = 0.89
    b = 0.6
    c = 0.28
    k = 0.04
    threshold = 1.0
    x = 0.0
    y = 0.0
    x_values: list[float] = []
    y_values: list[float] = []
    events: list[int] = []

    for _step in range(steps):
        x_previous = x
        x_next = x * x * math.exp(y - x) + k + current
        y_next = a * y - b * x + c
        x, y = x_next, y_next
        events.append(int(x_previous < threshold <= x))
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


def _independent_receipt(*, current: float, steps: int) -> dict[str, object]:
    """Derive byte-level state and event evidence without production code."""
    x = y = 0.0
    state_bytes = bytearray()
    event_bytes = bytearray()
    first_event: int | None = None
    for index in range(steps):
        x_previous = x
        x_next = x * x * math.exp(y - x) + 0.04 + current
        y_next = 0.89 * y - 0.6 * x + 0.28
        x, y = x_next, y_next
        event = int(x_previous < 1.0 <= x)
        if event and first_event is None:
            first_event = index
        state_bytes.extend(struct.pack("<dd", x, y))
        event_bytes.append(event)
    return {
        "events": sum(event_bytes),
        "first_event_index_zero_based": first_event,
        "final_state": {"x": x, "y": y},
        "state_trace_sha256": hashlib.sha256(state_bytes).hexdigest(),
        "event_trace_sha256": hashlib.sha256(event_bytes).hexdigest(),
    }


def test_features_match_independent_source_iteration() -> None:
    """Committed features must match a fresh Eq. 1 re-derivation."""
    spec = load_reference_trace_spec("chialvo_map_doi")
    expected = _independent_features(
        current=spec.protocol.inputs["I"],
        steps=spec.protocol.steps,
    )

    assert spec.schema_name == "chialvo_map"
    assert spec.provenance.kind == "map_iteration_reference"
    assert spec.provenance.citation == "doi:10.1016/0960-0779(93)E0056-H"
    assert expected["spike_count"] == 2.0
    assert expected["first_spike_step"] == 33.0
    assert set(expected) == set(spec.expected_features)
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)


def test_committed_trace_validates_through_schema_runner() -> None:
    """The production schema runner must reproduce the DOI feature contract."""
    spec = load_reference_trace_spec("chialvo_map_doi")
    report = validate_reference_trace_spec(spec)
    assert report.passed
    assert report.mismatches == ()


def test_full_state_and_event_receipt_matches_independent_iteration() -> None:
    receipt_path = (
        Path(__file__).resolve().parents[1]
        / "src/sc_neurocore/neurons/reference_receipts/chialvo_1995.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["equation_origin"]["citation"] == "doi:10.1016/0960-0779(93)E0056-H"
    expected = _independent_receipt(**receipt["drive"])
    assert receipt["oracle"] == expected
