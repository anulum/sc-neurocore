# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Generated foreign runtime state conformance matrix

"""The matrix says how much of each model's state a lane could carry.

Operators can inspect each model and runtime lane: which declared variables
would cross the boundary, which would not, and whether the lane exports anything the model has no name
for. These cases hold the committed file to the live contract, and hold the
matrix to the one question it answers — a lane's *contract*, never whether that
lane happens to be built on this machine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sc_neurocore.studio.runtime_state_packet import (
    RUNTIME_STATE_PACKET_SCHEMA_VERSION,
    RUST_BATCH_PACKET,
)
from sc_neurocore.studio.state_layout import declared_state
from tools.runtime_state_conformance import (
    CONFORMANCE_SCHEMA,
    DEFAULT_OUTPUT,
    LANES,
    build_matrix,
    encode,
    main,
    model_names,
    render_summary,
    selectable_models,
)

_COMMITTED = Path(__file__).resolve().parents[1] / DEFAULT_OUTPUT


@pytest.fixture(scope="module")
def matrix() -> dict[str, Any]:
    """Build the matrix once for the module."""
    return build_matrix()


def _rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    rows = matrix["rows"]
    assert isinstance(rows, list)
    return rows


class TestTheCommittedMatrix:
    """Check committed matrix bytes and deterministic source-derived encoding."""

    def test_the_committed_file_matches_the_live_contract(self) -> None:
        """The drift gate: a changed layout or packet must be regenerated."""
        assert _COMMITTED.is_file()

        assert _COMMITTED.read_text(encoding="utf-8") == encode(build_matrix())

    def test_the_file_records_no_timestamp_or_commit(self) -> None:
        """It changes when a layout or a packet changes, and at no other time."""
        text = _COMMITTED.read_text(encoding="utf-8")

        assert "generated_at" not in text
        assert "commit" not in text


class TestTheMatrix:
    """Check declared model state and packet accounting across every runtime lane."""

    def test_it_names_its_contract_and_the_packet_contract(self, matrix: dict[str, Any]) -> None:
        """Retain the matrix schema and each runtime packet contract."""
        assert matrix["schema_version"] == CONFORMANCE_SCHEMA
        assert matrix["packet_schema_version"] == RUNTIME_STATE_PACKET_SCHEMA_VERSION
        assert matrix["lanes"] == [RUST_BATCH_PACKET.to_public_dict()]

    def test_every_catalogue_model_has_a_row(self, matrix: dict[str, Any]) -> None:
        """Account for every catalogue model exactly once in the sorted matrix."""
        rows = _rows(matrix)

        assert [row["model"] for row in rows] == list(model_names())
        assert matrix["summary"]["models"] == len(rows)

    def test_every_row_is_judged_by_every_lane(self, matrix: dict[str, Any]) -> None:
        """Include every declared runtime lane in each model row."""
        for row in _rows(matrix):
            assert sorted(row["lanes"]) == sorted(packet.runtime for packet in LANES)

    def test_a_row_reports_the_model_s_own_declared_state(self, matrix: dict[str, Any]) -> None:
        """Preserve the model's declared state variable names in its row."""
        by_model = {row["model"]: row for row in _rows(matrix)}
        expected = [spec.name for spec in declared_state("PinskyRinzelNeuron")[2]]

        assert by_model["PinskyRinzelNeuron"]["declared"] == expected

    def test_a_model_the_lane_can_name_nothing_in_is_reported_as_such(
        self, matrix: dict[str, Any]
    ) -> None:
        """Report unnameable state without claiming complete transport."""
        by_model = {row["model"]: row for row in _rows(matrix)}
        coverage = by_model["PinskyRinzelNeuron"]["lanes"]["rust-batch"]

        assert coverage["carried"] == []
        assert coverage["unnameable"] == ["v"]
        assert coverage["complete"] is False

    def test_a_model_the_lane_partly_carries_is_reported_as_such(
        self, matrix: dict[str, Any]
    ) -> None:
        """Distinguish carried and dropped variables for partial transport."""
        by_model = {row["model"]: row for row in _rows(matrix)}
        coverage = by_model["AdExNeuron"]["lanes"]["rust-batch"]

        assert coverage["carried"] == ["v"]
        assert coverage["dropped"] == ["w"]
        assert coverage["complete"] is False

    def test_the_census_adds_up_to_the_rows(self, matrix: dict[str, Any]) -> None:
        """A summary nobody can re-derive from the rows is a claim, not a count."""
        rows = _rows(matrix)
        counts = matrix["summary"]["per_lane"]["rust-batch"]

        assert counts["carried"] == sum(len(row["lanes"]["rust-batch"]["carried"]) for row in rows)
        assert counts["dropped"] == sum(len(row["lanes"]["rust-batch"]["dropped"]) for row in rows)
        assert counts["names_nothing"] == sum(
            1 for row in rows if row["declared"] and not row["lanes"]["rust-batch"]["carried"]
        )
        assert counts["complete"] == sum(
            1 for row in rows if row["declared"] and row["lanes"]["rust-batch"]["complete"]
        )

    def test_models_without_declared_state_are_counted_apart(self, matrix: dict[str, Any]) -> None:
        """Separate undeclared model state from measured transport coverage."""
        rows = _rows(matrix)

        assert matrix["summary"]["models_without_declared_state"] == sum(
            1 for row in rows if not row["declared"]
        )


class TestTheSummary:
    """Check operator census output and malformed summary refusal."""

    def test_it_names_every_lane_and_its_counts(self, matrix: dict[str, Any]) -> None:
        """Print the declared variable and model counts for each runtime lane."""
        text = render_summary(matrix)

        assert "rust-batch" in text
        assert "declared variables carried" in text
        assert "models it can name nothing in" in text
        assert str(matrix["summary"]["models"]) in text

    def test_a_matrix_without_a_summary_mapping_is_refused(self) -> None:
        """Refuse an operator summary whose census is not a mapping."""
        with pytest.raises(TypeError, match="mapping for its summary"):
            render_summary({"schema_version": CONFORMANCE_SCHEMA, "summary": []})

    def test_a_matrix_without_a_lane_census_is_refused(self) -> None:
        """Refuse an operator summary with malformed per-lane accounting."""
        with pytest.raises(TypeError, match="mapping for its lane census"):
            render_summary(
                {
                    "schema_version": CONFORMANCE_SCHEMA,
                    "summary": {
                        "models": 0,
                        "models_without_declared_state": 0,
                        "per_lane": [],
                    },
                }
            )

    def test_a_malformed_selectable_census_is_refused(self, matrix: dict[str, Any]) -> None:
        """Reject malformed selectable counts instead of reporting a false census."""
        altered = dict(matrix)
        altered["summary"] = dict(matrix["summary"])
        altered["summary"]["per_lane_selectable"] = []

        with pytest.raises(TypeError, match="mapping for its selectable census"):
            render_summary(altered)

    def test_a_matrix_without_selectable_counts_retains_the_whole_census(
        self, matrix: dict[str, Any]
    ) -> None:
        """Keep whole-catalogue accounting usable when selectable counts are absent."""
        altered = dict(matrix)
        altered["summary"] = dict(matrix["summary"])
        del altered["summary"]["per_lane_selectable"]

        text = render_summary(altered)

        assert "rust-batch" in text
        assert "declared variables carried" in text
        assert "it can be selected for" not in text


class TestCommandLine:
    """Exercise matrix writing, drift checks and actual command output."""

    def test_writing_and_checking_agree(self, tmp_path: Path) -> None:
        """Write a real matrix file and accept its unchanged source contract."""
        destination = tmp_path / "matrix.json"

        assert main(["--write", "--output", str(destination)]) == 0
        assert main(["--check", "--output", str(destination)]) == 0

        document = json.loads(destination.read_text(encoding="utf-8"))
        assert document["schema_version"] == CONFORMANCE_SCHEMA

    def test_an_absent_file_fails_the_check(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Refuse a missing matrix file with an explicit command error."""
        assert main(["--check", "--output", str(tmp_path / "absent.json")]) == 1

        assert "is absent" in capsys.readouterr().err

    def test_a_drifted_file_fails_the_check(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Refuse stored matrix bytes that differ from the current contract."""
        destination = tmp_path / "matrix.json"
        destination.write_text('{"schema_version": "stale"}\n', encoding="utf-8")

        assert main(["--check", "--output", str(destination)]) == 1

        assert "has drifted" in capsys.readouterr().err

    def test_the_default_invocation_prints_the_census(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Print the operator census without creating an output file."""
        assert main(["--output", str(tmp_path / "unused.json")]) == 0

        printed = capsys.readouterr().out
        assert "foreign runtime state conformance" in printed
        assert not (tmp_path / "unused.json").exists()

    def test_writing_also_prints_the_census_when_asked(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Write the matrix and print the requested per-lane census."""
        destination = tmp_path / "matrix.json"

        assert main(["--write", "--summary", "--output", str(destination)]) == 0

        printed = capsys.readouterr().out
        assert "Wrote" in printed
        assert "rust-batch" in printed


class TestTheSelectableCensus:
    """The narrower census: what a lane does for the models it can be selected for.

    It used to live on the public page as a number nothing emitted and nothing
    checked, and its figures could not be reproduced from any definition in the
    code. It is generated now, from the same committed sources as the matrix.
    """

    def test_the_selectable_set_is_parsed_from_the_committed_catalogue(self) -> None:
        """The list is read as text, so the matrix needs no built engine."""
        catalogue = selectable_models()
        assert "AdEx" in catalogue
        assert "LapicqueNeuron" in catalogue
        assert len(catalogue) > 100

    def test_the_parsed_catalogue_matches_the_built_engine(self) -> None:
        """A committed list that has drifted from the binary would be a lie."""
        engine = pytest.importorskip("sc_neurocore_engine")
        runner = getattr(engine, "NetworkRunner", None)
        if runner is None or not hasattr(runner, "supported_models"):
            pytest.skip("the installed engine exposes no supported-model list")
        assert set(runner.supported_models()) == set(selectable_models())

    def test_every_lane_reports_a_selectable_census(self, matrix: dict[str, Any]) -> None:
        """A lane without one would leave the page quoting nothing again."""
        summary = matrix["summary"]
        assert set(summary["per_lane_selectable"]) == {packet.runtime for packet in LANES}

    def test_the_selectable_census_counts_only_selectable_rows(
        self, matrix: dict[str, Any]
    ) -> None:
        """Recomputed from the rows, so the census cannot drift from the matrix."""
        catalogue = selectable_models()
        for packet in LANES:
            counted = [
                row
                for row in matrix["rows"]
                if str(row["model"]) in catalogue
                or (str(row["model"]).endswith("Neuron") and str(row["model"])[:-6] in catalogue)
            ]
            with_layout = [row for row in counted if row["declared"]]
            census = matrix["summary"]["per_lane_selectable"][packet.runtime]
            assert census["models"] == len(counted)
            assert census["models_with_declared_state"] == len(with_layout)
            assert census["carried"] == sum(
                len(row["lanes"][packet.runtime]["carried"]) for row in with_layout
            )
            assert census["dropped"] == sum(
                len(row["lanes"][packet.runtime]["dropped"]) for row in with_layout
            )

    def test_the_selectable_census_is_no_larger_than_the_whole_catalogue(
        self, matrix: dict[str, Any]
    ) -> None:
        """A restriction that counted more than the whole would be reversed."""
        for packet in LANES:
            whole = matrix["summary"]["per_lane"][packet.runtime]
            restricted = matrix["summary"]["per_lane_selectable"][packet.runtime]
            assert restricted["models"] <= matrix["summary"]["models"]
            for key in ("carried", "dropped", "complete", "names_nothing"):
                assert restricted[key] <= whole[key]

    def test_the_summary_prints_both_censuses(self, matrix: dict[str, Any]) -> None:
        """An operator reading the tool sees the same two blocks the page quotes."""
        printed = render_summary(matrix)
        restricted = matrix["summary"]["per_lane_selectable"]["rust-batch"]
        assert f"of the {restricted['models']} it can be selected for" in printed
        assert f"declared variables carried: {restricted['carried']}" in printed

    def test_the_public_page_quotes_the_generated_numbers(self) -> None:
        """The page must not carry a figure the generator does not produce."""
        page = (
            Path(__file__).resolve().parents[1]
            / "docs"
            / "validation"
            / "runtime_state_conformance.md"
        ).read_text(encoding="utf-8")
        summary = build_matrix()["summary"]
        assert isinstance(summary, dict)
        selectable = summary["per_lane_selectable"]
        assert isinstance(selectable, dict)
        restricted = selectable["rust-batch"]
        assert isinstance(restricted, dict)
        for value in (
            restricted["models"],
            restricted["models_with_declared_state"],
            restricted["carried"],
            restricted["dropped"],
            restricted["names_nothing"],
        ):
            assert f"**{value}**" in page
