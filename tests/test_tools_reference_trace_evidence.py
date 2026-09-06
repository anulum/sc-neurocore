# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Reference-trace evidence tool

"""The evidence an operator can print without reading the code.

The tool exists so the honest boundary — which identities rest on a published
source, and whether a trace can fail at all — is reachable from a shell. These
cases drive its real entry point and require the fail-closed form to refuse
when a trace proves nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.reference_trace_evidence import build_evidence, main, render_report


@pytest.fixture(scope="module")
def evidence() -> dict[str, object]:
    """Build the corpus evidence once for the module."""
    return build_evidence()


class TestEvidence:
    def test_the_evidence_carries_both_surfaces(self, evidence: dict[str, object]) -> None:
        assert set(evidence) == {"adjudication", "negative_controls"}
        adjudication = evidence["adjudication"]
        controls = evidence["negative_controls"]
        assert isinstance(adjudication, dict)
        assert isinstance(controls, dict)
        assert adjudication["total"] == len(adjudication["rows"])  # type: ignore[arg-type]
        assert controls["uncontrolled_traces"] == []

    def test_the_evidence_survives_a_json_round_trip(self, evidence: dict[str, object]) -> None:
        assert json.loads(json.dumps(evidence)) == evidence

    def test_the_report_states_the_counts_and_the_blind_spots(
        self, evidence: dict[str, object]
    ) -> None:
        report = render_report(evidence)

        assert "independence adjudication" in report
        assert "project_formulation" in report
        assert "every trace fails under at least one control" in report
        assert "applied but undetected:" in report
        assert "mcculloch_pitts_1943_truth_table / unit_scale" in report


class TestMalformedEvidence:
    def test_a_corpus_with_no_blind_spots_omits_the_section(self) -> None:
        """The section exists to name blind spots, not to be printed empty."""
        clean = {
            "adjudication": {
                "counts": {"published_source": 1},
                "rows": [],
                "total": 1,
                "version": "v",
            },
            "negative_controls": {
                "outcomes": [
                    {
                        "mismatched_features": 2,
                        "mutation": "sign",
                        "name": "a_trace",
                        "reason": "the drive sign was flipped",
                        "status": "detected",
                    }
                ],
                "uncontrolled_traces": [],
                "version": "v",
            },
        }

        report = render_report(clean)

        assert "applied but undetected" not in report
        assert "every trace fails under at least one control" in report

    def test_an_uncontrolled_corpus_is_named_in_the_report(self) -> None:
        report = render_report(
            {
                "adjudication": {"counts": {}, "rows": [], "total": 0, "version": "v"},
                "negative_controls": {
                    "outcomes": [],
                    "uncontrolled_traces": ["a_trace_that_proves_nothing"],
                    "version": "v",
                },
            }
        )

        assert "UNCONTROLLED: a_trace_that_proves_nothing" in report

    def test_a_non_mapping_section_is_refused(self) -> None:
        with pytest.raises(TypeError, match="expected a mapping"):
            render_report({"adjudication": [], "negative_controls": {}})

    def test_a_non_list_outcome_set_is_refused(self) -> None:
        with pytest.raises(TypeError, match="expected a list"):
            render_report(
                {
                    "adjudication": {"counts": {}, "version": "v"},
                    "negative_controls": {
                        "outcomes": {},
                        "uncontrolled_traces": [],
                        "version": "v",
                    },
                }
            )


class TestCommandLine:
    def test_report_prints_and_succeeds(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--report"]) == 0

        printed = capsys.readouterr().out
        assert "negative controls" in printed

    def test_check_succeeds_while_every_trace_is_controlled(self) -> None:
        assert main(["--check"]) == 0

    def test_json_writes_a_sorted_document(self, tmp_path: Path) -> None:
        destination = tmp_path / "evidence.json"

        assert main(["--json", str(destination)]) == 0

        document = json.loads(destination.read_text(encoding="utf-8"))
        assert set(document) == {"adjudication", "negative_controls"}
        assert destination.read_text(encoding="utf-8").endswith("\n")

    def test_check_refuses_an_uncontrolled_corpus(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A trace no mutation can break must not report success."""
        import tools.reference_trace_evidence as evidence_tool

        def _uncontrolled() -> dict[str, object]:
            return {
                "adjudication": {"counts": {}, "rows": [], "total": 0, "version": "v"},
                "negative_controls": {
                    "outcomes": [],
                    "uncontrolled_traces": ["a_trace_that_proves_nothing"],
                    "version": "v",
                },
            }

        monkeypatch.setattr(evidence_tool, "build_evidence", _uncontrolled)

        assert main(["--check"]) == 1
        assert "a_trace_that_proves_nothing" in capsys.readouterr().err

    def test_an_unadjudicated_corpus_is_reported_not_crashed(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import tools.reference_trace_evidence as evidence_tool
        from sc_neurocore.neurons.reference_trace_provenance import (
            ReferenceTraceAdjudicationError,
        )

        def _refuse() -> dict[str, object]:
            raise ReferenceTraceAdjudicationError("kind 'invented' is not adjudicated")

        monkeypatch.setattr(evidence_tool, "build_evidence", _refuse)

        assert main(["--report"]) == 1
        assert "not adjudicated" in capsys.readouterr().err
