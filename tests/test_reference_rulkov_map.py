# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rulkov map independent reference contract

"""Independent piecewise-map reference trace for the Rulkov map."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

from sc_neurocore.neurons.models.rulkov_map import RulkovMapNeuron
from sc_neurocore.neurons.models.sc_upward_crossing_rulkov_map import (
    SCUpwardCrossingRulkovMapNeuron,
)
from sc_neurocore.neurons.reference_traces import load_reference_trace_spec
from tests.cosim_support import _rulkov_map_features


_RECEIPT_ROOT = Path(__file__).parents[1] / "src/sc_neurocore/neurons/reference_receipts"


def _independent_receipt(*, source_event: bool) -> dict[str, float | int | str]:
    """Re-derive the mixed-drive receipt directly from Rulkov's equations."""
    pattern = (0.0, 0.25, 1.5, -0.1)
    x = -1.0
    y = -3.0
    alpha = 4.0
    sigma = -1.6
    mu = 0.001
    events = 0
    first_event = -1
    digest = hashlib.sha256()
    for index in range(512):
        current = pattern[index % len(pattern)]
        boundary = alpha + y + current
        reset_branch_event = int(x > 0.0 and x >= boundary)
        x_previous = x
        if x <= 0.0:
            x_next = alpha / (1.0 - x) + y + current
        elif x < boundary:
            x_next = boundary
        else:
            x_next = -1.0
        y_next = y - mu * (x + 1.0) + mu * sigma
        event = reset_branch_event if source_event else int(x_next >= 0.0 and x_previous < 0.0)
        x, y = x_next, y_next
        if event and first_event < 0:
            first_event = index
        events += event
        digest.update(struct.pack("<ddB", x, y, event))
    return {
        "events": events,
        "first_event_index": first_event,
        "x_final": x,
        "y_final": y,
        "trace_sha256": digest.hexdigest(),
    }


def _public_receipt(
    model: RulkovMapNeuron, pattern: tuple[float, ...]
) -> dict[str, float | int | str]:
    """Digest the public per-step state and event surface."""
    events = 0
    first_event = -1
    digest = hashlib.sha256()
    for index in range(512):
        event = model.step(pattern[index % len(pattern)])
        if event and first_event < 0:
            first_event = index
        events += event
        digest.update(struct.pack("<ddB", model.x, model.y, event))
    return {
        "events": events,
        "first_event_index": first_event,
        "x_final": model.x,
        "y_final": model.y,
        "trace_sha256": digest.hexdigest(),
    }


def test_rulkov_map_trace_features_match_independent_map_iteration() -> None:
    """Committed Rulkov features must match an independent piecewise-map iteration."""
    spec = load_reference_trace_spec("rulkov_map_driven_spiking_doi")

    expected = _rulkov_map_features(
        current=spec.protocol.inputs["I"],
        steps=spec.protocol.steps,
    )

    assert spec.schema_name == "rulkov_map"
    assert spec.provenance.kind == "map_iteration_reference"
    assert spec.provenance.citation == "doi:10.1103/PhysRevE.65.041922"
    assert spec.expected_features["spike_count"] > 0
    assert set(expected) == set(spec.expected_features)
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)


@pytest.mark.parametrize(
    ("receipt_name", "source_event", "model_type"),
    [
        ("rulkov_2002.json", True, RulkovMapNeuron),
        (
            "sc_upward_crossing_rulkov_project.json",
            False,
            SCUpwardCrossingRulkovMapNeuron,
        ),
    ],
)
def test_complete_receipt_matches_independent_equations_and_public_model(
    receipt_name: str,
    source_event: bool,
    model_type: type[RulkovMapNeuron],
) -> None:
    """Bind both 512-step receipts to an independent oracle and public model."""
    receipt = json.loads((_RECEIPT_ROOT / receipt_name).read_text(encoding="utf-8"))
    pattern = tuple(float(value) for value in receipt["drive"]["pattern"])
    assert pattern == (0.0, 0.25, 1.5, -0.1)
    assert int(receipt["drive"]["repeats"]) == 128
    assert receipt["oracle"] == _independent_receipt(source_event=source_event)
    assert receipt["oracle"] == _public_receipt(model_type(), pattern)
