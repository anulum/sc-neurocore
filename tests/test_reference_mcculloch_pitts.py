# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Independent McCulloch-Pitts 1943 reference evidence

"""Re-derive the published logical rule without production helper reuse."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

from sc_neurocore.neurons.models.mcculloch_pitts import McCullochPittsNeuron
from sc_neurocore.neurons.reference_traces import (
    reference_trace_spec_from_payload,
    validate_reference_trace_spec,
)

_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "src/sc_neurocore/neurons/reference_trace_data/mcculloch_pitts_1943_truth_table.json"
)
_ORACLE_SHA256 = "2aebd2a5ed6ea8a9409b7452d603441d91e2c8b610da8446668452492ee73db4"


def _payload() -> dict[str, object]:
    """Load the committed source-bound truth-table payload."""
    return cast(dict[str, object], json.loads(_ARTIFACT.read_text(encoding="utf-8")))


def _rows(payload: dict[str, object]) -> list[list[object]]:
    """Return the authored truth rows after their structural checks."""
    truth_table = cast(dict[str, object], payload["truth_table"])
    assert truth_table["columns"] == [
        "theta",
        "excitatory_count",
        "inhibitory_active",
        "output",
    ]
    rows = cast(list[list[object]], truth_table["rows"])
    assert all(len(row) == 4 for row in rows)
    return rows


def test_truth_rows_match_an_independent_absolute_veto_rule() -> None:
    """Every committed row follows the 1943 count threshold and absolute veto."""
    payload = _payload()
    for theta, excitatory_count, inhibitory_active, expected in _rows(payload):
        independent = int(
            not cast(bool, inhibitory_active) and cast(int, excitatory_count) >= cast(int, theta)
        )
        assert independent == expected
        assert (
            McCullochPittsNeuron(theta=cast(int, theta)).step(
                cast(int, excitatory_count),
                cast(bool, inhibitory_active),
            )
            == expected
        )


def test_truth_rows_have_a_stable_source_evidence_digest() -> None:
    """The descriptor can bind one canonical byte representation of the rule."""
    payload = _payload()
    lines = ["theta excitatory_count inhibitory_active output\n"]
    for theta, count, inhibited, output in _rows(payload):
        lines.append(f"{theta} {count} {int(cast(bool, inhibited))} {output}\n")
    digest = hashlib.sha256("".join(lines).encode()).hexdigest()
    provenance = cast(dict[str, object], payload["provenance"])
    assert digest == provenance["oracle_rows_sha256"] == _ORACLE_SHA256
    assert provenance["citation"] == "doi:10.1007/BF02478259"


def test_standard_corpus_runner_validates_stateless_activity() -> None:
    """The ordinary UniversalNeuron harness accepts event-only deterministic traces."""
    spec = reference_trace_spec_from_payload(_payload())
    report = validate_reference_trace_spec(spec)
    assert spec.schema_name == "mcculloch_pitts"
    assert spec.protocol.state_variables == ()
    assert report.passed
    assert report.mismatches == ()
    assert dict(report.simulation.trace) == {}
