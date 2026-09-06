# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Generated foreign runtime state conformance matrix

"""The matrix says how much of each model's state a lane could carry.

It is the answer to FF-05's question in a form an operator can read without the
code: per model and lane, which declared variables would cross the boundary,
which would not, and whether the lane exports anything the model has no name
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
    def test_it_names_its_contract_and_the_packet_contract(self, matrix: dict[str, Any]) -> None:
        assert matrix["schema_version"] == CONFORMANCE_SCHEMA
        assert matrix["packet_schema_version"] == RUNTIME_STATE_PACKET_SCHEMA_VERSION
        assert matrix["lanes"] == [RUST_BATCH_PACKET.to_public_dict()]

    def test_every_catalogue_model_has_a_row(self, matrix: dict[str, Any]) -> None:
        rows = _rows(matrix)

        assert [row["model"] for row in rows] == list(model_names())
        assert matrix["summary"]["models"] == len(rows)

    def test_every_row_is_judged_by_every_lane(self, matrix: dict[str, Any]) -> None:
        for row in _rows(matrix):
            assert sorted(row["lanes"]) == sorted(packet.runtime for packet in LANES)

    def test_a_row_reports_the_model_s_own_declared_state(self, matrix: dict[str, Any]) -> None:
        by_model = {row["model"]: row for row in _rows(matrix)}
        expected = [spec.name for spec in declared_state("PinskyRinzelNeuron")[2]]

        assert by_model["PinskyRinzelNeuron"]["declared"] == expected

    def test_a_model_the_lane_can_name_nothing_in_is_reported_as_such(
        self, matrix: dict[str, Any]
    ) -> None:
        by_model = {row["model"]: row for row in _rows(matrix)}
        coverage = by_model["PinskyRinzelNeuron"]["lanes"]["rust-batch"]

        assert coverage["carried"] == []
        assert coverage["unnameable"] == ["v"]
        assert coverage["complete"] is False

    def test_a_model_the_lane_partly_carries_is_reported_as_such(
        self, matrix: dict[str, Any]
    ) -> None:
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
        rows = _rows(matrix)

        assert matrix["summary"]["models_without_declared_state"] == sum(
            1 for row in rows if not row["declared"]
        )


class TestTheSummary:
    def test_it_names_every_lane_and_its_counts(self, matrix: dict[str, Any]) -> None:
        text = render_summary(matrix)

        assert "rust-batch" in text
        assert "declared variables carried" in text
        assert "models it can name nothing in" in text
        assert str(matrix["summary"]["models"]) in text

    def test_a_matrix_without_a_summary_mapping_is_refused(self) -> None:
        with pytest.raises(TypeError, match="mapping for its summary"):
            render_summary({"schema_version": CONFORMANCE_SCHEMA, "summary": []})

    def test_a_matrix_without_a_lane_census_is_refused(self) -> None:
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


class TestCommandLine:
    def test_writing_and_checking_agree(self, tmp_path: Path) -> None:
        destination = tmp_path / "matrix.json"

        assert main(["--write", "--output", str(destination)]) == 0
        assert main(["--check", "--output", str(destination)]) == 0

        document = json.loads(destination.read_text(encoding="utf-8"))
        assert document["schema_version"] == CONFORMANCE_SCHEMA

    def test_an_absent_file_fails_the_check(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["--check", "--output", str(tmp_path / "absent.json")]) == 1

        assert "is absent" in capsys.readouterr().err

    def test_a_drifted_file_fails_the_check(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        destination = tmp_path / "matrix.json"
        destination.write_text('{"schema_version": "stale"}\n', encoding="utf-8")

        assert main(["--check", "--output", str(destination)]) == 1

        assert "has drifted" in capsys.readouterr().err

    def test_the_default_invocation_prints_the_census(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["--output", str(tmp_path / "unused.json")]) == 0

        printed = capsys.readouterr().out
        assert "foreign runtime state conformance" in printed
        assert not (tmp_path / "unused.json").exists()

    def test_writing_also_prints_the_census_when_asked(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        destination = tmp_path / "matrix.json"

        assert main(["--write", "--summary", "--output", str(destination)]) == 0

        printed = capsys.readouterr().out
        assert "Wrote" in printed
        assert "rust-batch" in printed
