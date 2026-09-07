# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Benchmark coverage is counted, not asserted

"""Catalogue benchmark coverage is counted, and apparatus is not evidence.

The 7-point audit carries "Benchmarks FAIL — 63/173 covered (36%)" as a
hand-written row that nothing computes. Measured against live sources the figure
matches neither of the two things it could mean: 101 of 185 models are *named by*
a benchmark script, and 28 have a *committed record* naming them. The 73 models
between those figures own the apparatus and show no evidence anyone ran it.

These cases hold the census to that distinction. A script is not a measurement,
the three states stay exhaustive and disjoint, and the totals are pinned so the
gap shrinks only by a deliberate test change — never by a scanner quietly
counting scripts as results again.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.benchmark_coverage_census import (
    CENSUS_SCHEMA,
    DEFAULT_OUTPUT,
    STATES,
    build_census,
    encode,
    render_summary,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The measured census at the time this was pinned. It shrinks only by a
#: deliberate change here, and `apparatus_only` shrinking while `measured` does
#: not would mean scripts were deleted rather than benchmarks run.
PINNED_TOTALS = {"measured": 28, "apparatus_only": 73, "absent": 84}


@pytest.fixture(scope="module")
def census() -> dict[str, object]:
    """Build the census once from committed sources."""
    return build_census()


def _rows(census: dict[str, object]) -> list[dict[str, object]]:
    rows = census["rows"]
    assert isinstance(rows, list)
    return rows


class TestTheCommittedCensusIsCurrent:
    def test_the_file_matches_the_tool(self, census: dict[str, object]) -> None:
        """A census read from a stale file answers about a repository that moved."""
        committed = (REPO_ROOT / DEFAULT_OUTPUT).read_text(encoding="utf-8")
        assert committed == encode(census)

    def test_it_declares_its_schema(self, census: dict[str, object]) -> None:
        """A reader must be able to tell which contract produced the numbers."""
        assert census["schema_version"] == CENSUS_SCHEMA


class TestTheStatesPartitionTheCatalogue:
    def test_every_model_has_exactly_one_state(self, census: dict[str, object]) -> None:
        """A model counted twice, or not at all, makes the totals meaningless."""
        rows = _rows(census)
        assert len(rows) == census["catalogue_models"]
        assert {str(row["state"]) for row in rows} <= set(STATES)
        assert len({str(row["model"]) for row in rows}) == len(rows)

    def test_the_totals_are_the_row_counts(self, census: dict[str, object]) -> None:
        """Totals computed apart from the rows can drift from them."""
        totals = census["totals"]
        assert isinstance(totals, dict)
        for state in STATES:
            assert totals[state] == sum(1 for row in _rows(census) if row["state"] == state)
        assert sum(int(totals[state]) for state in STATES) == census["catalogue_models"]


class TestApparatusIsNotEvidence:
    def test_the_two_are_distinguished_and_the_gap_is_real(self, census: dict[str, object]) -> None:
        """The whole point: owning a benchmark script is not having been measured."""
        totals = census["totals"]
        assert isinstance(totals, dict)
        assert int(totals["apparatus_only"]) > 0

    def test_an_apparatus_only_model_has_a_script_and_no_record(
        self, census: dict[str, object]
    ) -> None:
        """The state must mean what it says for every row carrying it."""
        for row in _rows(census):
            if row["state"] == "apparatus_only":
                assert row["scripts"]
                assert row["records"] == []

    def test_a_measured_model_is_named_by_the_record_it_cites(
        self, census: dict[str, object]
    ) -> None:
        """A citation that does not name the model is not evidence for it."""
        for row in _rows(census):
            if row["state"] != "measured":
                continue
            records = row["records"]
            assert isinstance(records, list) and records
            for name in records:
                text = (REPO_ROOT / "benchmarks" / "results" / str(name)).read_text(
                    encoding="utf-8"
                )
                assert str(row["model"]) in text

    def test_an_absent_model_has_neither(self, census: dict[str, object]) -> None:
        """Otherwise 'absent' would understate what exists."""
        for row in _rows(census):
            if row["state"] == "absent":
                assert row["records"] == []
                assert row["scripts"] == []


class TestTheGapIsPinned:
    def test_the_totals_are_the_pinned_ones(self, census: dict[str, object]) -> None:
        """Pinned rather than sampled: it moves only by a deliberate change here."""
        totals = census["totals"]
        assert isinstance(totals, dict)
        assert {state: int(totals[state]) for state in STATES} == PINNED_TOTALS

    def test_the_summary_states_the_distinction(self, census: dict[str, object]) -> None:
        """An operator reading only the summary must not mistake one for the other."""
        summary = render_summary(census)
        assert "apparatus" in summary
        assert "not a measurement" in summary


class TestItAnswersADifferentQuestionFromTheEvidenceGate:
    def test_the_census_output_is_not_the_gate_output(self) -> None:
        """Quality of the artefacts that exist, and how many models have any."""
        assert "benchmark_coverage_census" in DEFAULT_OUTPUT
        gate = REPO_ROOT / "tools" / "benchmark_evidence_gate.py"
        assert gate.is_file()
        assert "coverage_census" not in gate.read_text(encoding="utf-8")
