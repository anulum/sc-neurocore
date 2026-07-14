# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Independent IQIF source-implementation reference

"""Re-derive the pinned C++ tutorial without importing production model code."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sc_neurocore.neurons.models.iqif import IntegerQIFNeuron
from sc_neurocore.neurons.reference_traces import (
    reference_trace_spec_from_payload,
    validate_reference_trace_spec,
)

_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "src/sc_neurocore/neurons/reference_trace_data/iqif_a8752eb_tutorial.json"
)
_ORACLE_SHA256 = "57c6916aac726033546610a2e784a22cf5b57d70657779abc7269510e9e64bf3"


def _independent_source_rows() -> tuple[list[int], list[int], bytes]:
    """Evaluate the literal pinned-source recurrence and serialise its rows."""
    rest = 128
    threshold = 200
    reset = 128
    a = 1
    b = 1
    v_max = 255
    v_min = 0
    current = 10
    branch_point = (b * threshold + a * rest) // (a + b)
    v = rest
    values: list[int] = []
    events: list[int] = []
    rows: list[str] = []
    for index in range(400):
        force = a * (rest - v) if v < branch_point else b * (v - threshold)
        v += (force >> 3) + current
        event = int(v > v_max)
        if event:
            v = reset
        elif v < v_min:
            v = v_min
        values.append(v)
        events.append(event)
        rows.append(f"{index} {event} {v}\n")
    return values, events, "".join(rows).encode()


def test_source_trace_and_runtime_features_are_exact() -> None:
    """Independent recurrence, source text and public Python trace are identical."""
    values, events, source_text = _independent_source_rows()
    assert hashlib.sha256(source_text).hexdigest() == _ORACLE_SHA256
    assert sum(events) == 26
    assert next(index for index, event in enumerate(events, start=1) if event) == 15
    assert (values[-1], min(values), max(values), sum(values) / len(values)) == (
        198,
        128,
        242,
        179.76,
    )

    public = IntegerQIFNeuron()
    trace, spikes = public.simulate(400, 10, backend="python")
    assert trace.tolist() == values
    assert spikes == sum(events)


def test_committed_provenance_binds_exact_source_commit_and_hashes() -> None:
    """The corpus names the immutable code source, paper and oracle bytes."""
    payload = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
    provenance = payload["provenance"]
    assert provenance["kind"] == "source_implementation_reference"
    assert provenance["source_commit"] == "a8752eba49dba9ba43a64be74090b91a51044b2f"
    assert provenance["source_cpp_sha256"] == (
        "1c7fb3184a82a1fdd8a2c29a7420da62987fce953bf8ce33a5b08fea2f880b99"
    )
    assert provenance["source_header_sha256"] == (
        "4beac58c1685cb19332ec6067bfa5eb2ce3c15d9196ba6ddd6aaa2ca608f9de1"
    )
    assert provenance["oracle_text_sha256"] == _ORACLE_SHA256
    assert provenance["citation"] == "doi:10.1109/AICAS51828.2021.9458572"


def test_committed_trace_validates_through_schema_runner() -> None:
    """The standard corpus runner reproduces every zero-tolerance feature."""
    payload = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
    spec = reference_trace_spec_from_payload(payload)
    assert spec.schema_name == "iqif"
    assert spec.provenance.kind == "source_implementation_reference"
    report = validate_reference_trace_spec(spec)
    assert report.passed
    assert report.mismatches == ()
    assert report.simulation.features == spec.expected_features
