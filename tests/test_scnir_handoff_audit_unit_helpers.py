# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (unit_helpers) from former test_scnir_handoff_audit.py

from __future__ import annotations

from tests.scnir_handoff_audit_support import *  # noqa: F403


def test_source_row_match_accepts_lfsr_fields() -> None:
    """A matching LFSR source row carries seed, polynomial and tap-mask expectations."""
    _verify_source_row_matches_stream(_lfsr_row(), _lfsr_stream(), 0)


def test_source_row_match_rejects_lfsr_tap_mask_mismatch() -> None:
    """A divergent LFSR tap mask is reported against the stream value."""
    row = _lfsr_row()
    row["tap_mask"] = 0x1234

    with pytest.raises(SCNIRHDLHandoffAuditError, match="tap_mask"):
        _verify_source_row_matches_stream(row, _lfsr_stream(), 0)


def test_delay_steps_for_row_expands_vector() -> None:
    """A per-source-column delay vector is materialised as a list of ints."""
    assert _delay_steps_for_row((1, 2, 3)) == [1, 2, 3]


def test_low_level_manifest_validators_reject_bad_values() -> None:
    """The structural manifest validators reject malformed sequences and scalars."""
    with pytest.raises(SCNIRHDLHandoffAuditError, match="must be a sequence"):
        _expect_mapping_sequence({"rows": "nope"}, "rows")
    with pytest.raises(SCNIRHDLHandoffAuditError, match="must be a JSON object"):
        _expect_mapping_sequence({"rows": [123]}, "rows")
    with pytest.raises(SCNIRHDLHandoffAuditError, match="must be a non-empty string"):
        _expect_non_empty_string({"k": ""}, "k")
    with pytest.raises(SCNIRHDLHandoffAuditError, match="must be an integer"):
        _expect_int({"k": "x"}, "k")
    with pytest.raises(SCNIRHDLHandoffAuditError, match="must be positive"):
        _expect_positive_int({"k": 0}, "k")
    with pytest.raises(SCNIRHDLHandoffAuditError, match="must be non-negative"):
        _expect_non_negative_int({"k": -1}, "k")


def test_write_audit_report_emits_valid_json(tmp_path: Path) -> None:
    """Writing the audit report serialises the valid summary to JSON on disk."""
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    output = tmp_path / "audit.json"

    report = write_scnir_hdl_handoff_audit(handoff, output)

    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["status"] == "valid"
    assert written["module_name"] == report.module_name
    assert written == report.as_dict()
