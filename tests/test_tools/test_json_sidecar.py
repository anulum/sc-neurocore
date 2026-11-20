# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Ownership sidecars written and checked for generated JSON

"""Hold the generated JSON sidecar to the ownership contract and to the committed sidecars."""

from __future__ import annotations

from pathlib import Path

from tools.json_sidecar import (
    COPYRIGHT_RECORDS,
    LICENCE,
    sidecar_path,
    sidecar_problems,
    sidecar_text,
    write_sidecar,
)

ROOT = Path(__file__).resolve().parents[2]


def test_the_sidecar_sits_next_to_its_json_file() -> None:
    """The sidecar is the JSON path with ``.license`` appended."""
    assert sidecar_path(Path("a/b/report.json")) == Path("a/b/report.json.license")


def test_the_text_is_the_project_header_then_both_copyright_records() -> None:
    """Seven header comments, a blank line, and the two records the contract names."""
    lines = sidecar_text("a measured record").splitlines()
    assert len(lines) == 10
    assert lines[0] == f"# {LICENCE}"
    assert lines[6] == "# SC-NeuroCore — REUSE sidecar for a measured record"
    assert lines[7] == ""
    assert lines[8:] == [f"# {record}" for record in COPYRIGHT_RECORDS]


def test_the_header_matches_a_committed_fixture_sidecar() -> None:
    """Apart from the title, the generated header is the one the fixture sidecars carry."""
    committed = (ROOT / "tests/fixtures/dense_layer_forward_golden.json.license").read_text(
        encoding="utf-8"
    )
    generated = sidecar_text("anything")
    assert generated.splitlines()[:6] == committed.splitlines()[:6]
    assert generated.splitlines()[7:] == committed.splitlines()[7:]


def test_write_then_check_and_each_way_a_sidecar_can_be_wrong(tmp_path: Path) -> None:
    """A missing or drifted sidecar is a problem; the written one is not."""
    report = tmp_path / "report.json"
    report.write_text("{}\n", encoding="utf-8")
    assert sidecar_problems(report, "a report") == [
        "no ownership sidecar report.json.license; run --write"
    ]
    written = write_sidecar(report, "a report")
    assert written == sidecar_path(report)
    assert sidecar_problems(report, "a report") == []
    assert sidecar_problems(report, "another report") == [
        "the ownership sidecar report.json.license has drifted; run --write"
    ]


def test_the_committed_generated_reports_carry_the_sidecars_their_tools_write() -> None:
    """Both behaviour reports' committed sidecars equal what their generators write."""
    from tools.model_behaviour_report import DEFAULT_OUTPUT as REPORT
    from tools.model_behaviour_report import SIDECAR_SUBJECT as REPORT_SUBJECT
    from tools.model_behaviour_variants import DEFAULT_OUTPUT as VARIANTS
    from tools.model_behaviour_variants import SIDECAR_SUBJECT as VARIANTS_SUBJECT

    assert sidecar_problems(ROOT / REPORT, REPORT_SUBJECT) == []
    assert sidecar_problems(ROOT / VARIANTS, VARIANTS_SUBJECT) == []
