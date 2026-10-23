# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public source provenance binding contracts

"""Verify actual native provenance persistence, independent checksums and stale-source refusal."""

from __future__ import annotations

import hashlib
import json

import pytest

from tests.engine_requirement import require_engine

require_engine()

from sc_neurocore_engine.ir import ScGraphBuilder, parse_ir


def _subject() -> ScGraphBuilder:
    b = ScGraphBuilder("source_subject")
    state = b.state(b.constant_i64(3, "i16"), "i16")
    value = b.input("sample", "i16")
    b.store(state, value)
    b.output("result", state)
    return b


@pytest.mark.parametrize(
    "payload", ["abc", '{"equation":"dv/dt = -v","author":"Šotek"}\n', "line1\r\nline2\t\0"]
)
def test_source_payload_roundtrips_with_independent_hashes_and_identical_hardware(
    payload: str,
) -> None:
    """Actual native text preserves exact UTF-8 bytes while both hardware outputs stay identical."""
    plain = _subject().build()
    b = _subject()
    b.bind_source(payload)
    graph = b.build()
    assert graph.verify() is None
    restored = parse_ir(graph.to_text())
    assert restored.to_text() == graph.to_text()
    assert restored.source_payload == payload
    assert restored.source_sha256 == hashlib.sha256(payload.encode()).hexdigest()
    assert restored.source_graph_sha256 == hashlib.sha256(plain.to_text().encode()).hexdigest()
    assert restored.emit_sv() == plain.emit_sv()
    assert restored.emit_mlir() == plain.emit_mlir()
    assert plain.source_payload is None
    assert plain.source_sha256 is None
    assert plain.source_graph_sha256 is None


@pytest.mark.parametrize("field", ["equation", "method", "initial", "dt", "rounding"])
def test_distinct_source_contract_fields_remain_distinct_in_native_ir(field: str) -> None:
    """Source differences survive even when two submitted numeric graphs are equivalent."""
    first = _subject()
    first.bind_source(json.dumps({field: "first"}, sort_keys=True))
    second = _subject()
    second.bind_source(json.dumps({field: "second"}, sort_keys=True))
    a, b = first.build(), second.build()
    assert a.source_sha256 != b.source_sha256
    assert a.to_text() != b.to_text()
    assert parse_ir(a.to_text()).source_payload != parse_ir(b.to_text()).source_payload
    assert a.emit_sv() == b.emit_sv()


def test_empty_duplicate_consumed_and_late_source_binding_refuse_without_silent_rebinding() -> None:
    """Actual builder errors preserve previous data; later operations invalidate both emitters."""
    b = _subject()
    with pytest.raises(ValueError, match="empty"):
        b.bind_source("")
    b.bind_source("abc")
    with pytest.raises(ValueError, match="already bound"):
        b.bind_source("replacement")
    graph = b.build()
    assert graph.source_payload == "abc"
    assert graph.verify() is None
    with pytest.raises(ValueError, match="consumed"):
        b.bind_source("replacement")
    b = _subject()
    b.bind_source("abc")
    b.input("late_input", "bool")
    stale = b.build()
    assert stale.verify()
    with pytest.raises(ValueError, match="graph checksum"):
        stale.emit_sv()
    with pytest.raises(ValueError, match="graph checksum"):
        stale.emit_mlir()
    with pytest.raises(ValueError, match="graph checksum"):
        parse_ir(stale.to_text())


@pytest.mark.parametrize("change", ["name", "constant", "payload", "version", "unknown"])
def test_changed_graph_or_source_text_cannot_be_imported_as_a_valid_binding(change: str) -> None:
    """Native parser rejects actual stale or malformed serialized provenance."""
    b = _subject()
    b.bind_source("abc")
    text = b.build().to_text()
    changed = {
        "name": text.replace("@source_subject", "@other_subject"),
        "constant": text.replace("sc.constant 3", "sc.constant 4"),
        "payload": text.replace("payload=616263", "payload=616264"),
        "version": text.replace("sc.source v1", "sc.source v2"),
        "unknown": text.replace("payload=616263", "payload=616263 unknown=1"),
    }[change]
    assert changed != text
    with pytest.raises(ValueError):
        parse_ir(changed)
