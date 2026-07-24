# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (rows_and_schema) from former test_readiness_evidence_index.py

from __future__ import annotations

from readiness_evidence_index_support import *  # noqa: F403

def test_build_rows_returns_one_row_per_enrolled(tool: ModuleType) -> None:
    """Inventory length matches the curated enrolment table."""
    rows = tool.build_rows()
    assert len(rows) == len(tool.ENROLLED)
    names = {row.class_name for row in rows}
    assert "AdExNeuron" in names
    assert "EscapeRateNeuron" in names
    assert "PoissonNeuron" in names
    assert "WangBuzsakiNeuron" in names

def test_validation_and_silicon_sections_are_honest_for_h1(tool: ModuleType) -> None:
    """H1 entries claim parity metric and cosim anchors; H0 does not claim cosim."""
    h1 = next(e for e in tool.ENROLLED if e.level == "h1_cosim" and not e.skip_apply)
    h0 = next(e for e in tool.ENROLLED if e.level == "h0_compile")
    v1 = tool.validation_section(h1, has_dynamics=True)
    s1 = tool.silicon_section(h1)
    assert v1["dynamics_faithful"] is True
    assert v1["metric"] == "parity"
    assert v1["evidence"]
    assert s1["compiles"] is True
    assert s1["cosim_validated"] is True
    assert s1["target_tier"] == "H1"

    v0 = tool.validation_section(h0, has_dynamics=True)
    s0 = tool.silicon_section(h0)
    assert v0["metric"] == "none"
    assert s0["compiles"] is True
    assert s0["cosim_validated"] is False
    assert s0["target_tier"] == "H0"
    v0_empty = tool.validation_section(h0, has_dynamics=False)
    assert v0_empty["dynamics_faithful"] is False

def test_index_payload_schema_and_json_roundtrip(tool: ModuleType, tmp_path: Path) -> None:
    """JSON export is valid and carries the schema version marker."""
    rows = tool.build_rows()
    payload = tool.index_payload(rows)
    assert payload["schema_version"] == tool.SCHEMA_VERSION
    assert payload["enrolled_count"] == len(tool.ENROLLED)
    out = tmp_path / "index.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == tool.SCHEMA_VERSION
    assert len(loaded["rows"]) == len(rows)

def test_main_report_exits_zero(tool: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI --report runs against the live corpus without error."""
    code = tool.main(["--report"])
    captured = capsys.readouterr()
    assert code == 0
    assert "AdExNeuron" in captured.out
    assert "open_gaps=" in captured.out
