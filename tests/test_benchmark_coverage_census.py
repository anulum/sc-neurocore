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

import hashlib
import json
from pathlib import Path
import runpy
import sys

import pytest

from tools.benchmark_coverage_census import (
    CENSUS_SCHEMA,
    CURRENCIES,
    DEFAULT_OUTPUT,
    STATES,
    apparatus_models,
    build_census,
    encode,
    main,
    measured_models,
    record_currency,
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


def _records(row: dict[str, object]) -> list[dict[str, object]]:
    records = row["records"]
    assert isinstance(records, list)
    return records


class TestTheCommittedCensusIsCurrent:
    """The committed census equals what the tool computes now."""

    def test_the_file_matches_the_tool(self, census: dict[str, object]) -> None:
        """A census read from a stale file answers about a repository that moved."""
        committed = (REPO_ROOT / DEFAULT_OUTPUT).read_text(encoding="utf-8")
        assert committed == encode(census)

    def test_it_declares_its_schema(self, census: dict[str, object]) -> None:
        """A reader must be able to tell which contract produced the numbers."""
        assert census["schema_version"] == CENSUS_SCHEMA


class TestTheStatesPartitionTheCatalogue:
    """Every model has exactly one of the three states."""

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
    """A script that names a model is kept apart from a record."""

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
            for record in records:
                assert isinstance(record, dict)
                text = (REPO_ROOT / "benchmarks" / "results" / str(record["name"])).read_text(
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
    """The model totals move only by a deliberate change."""

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
    """The census counts; the evidence gate judges."""

    def test_the_census_output_is_not_the_gate_output(self) -> None:
        """Quality of the artefacts that exist, and how many models have any."""
        assert "benchmark_coverage_census" in DEFAULT_OUTPUT
        gate = REPO_ROOT / "tools" / "benchmark_evidence_gate.py"
        assert gate.is_file()
        assert "coverage_census" not in gate.read_text(encoding="utf-8")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_with_one_record(root: Path, *, folder: str = "") -> tuple[Path, Path]:
    """Write a source and a record that binds it by path; return both paths."""
    source = root / "src" / "kernel.py"
    source.parent.mkdir(parents=True)
    source.write_text("STEP = 1\n", encoding="utf-8")
    untouched = root / "src" / "helper.py"
    untouched.write_text("HELP = 2\n", encoding="utf-8")
    record = root / "benchmarks" / "results" / folder / "bench_adex.json"
    record.parent.mkdir(parents=True)
    record.write_text(
        json.dumps(
            {
                "model": "AdExNeuron",
                "source_hashes": {
                    "src/kernel.py": _digest(source),
                    "src/helper.py": _digest(untouched),
                },
            }
        ),
        encoding="utf-8",
    )
    return source, record


def _adex_row(census: dict[str, object]) -> dict[str, object]:
    return next(row for row in _rows(census) if row["model"] == "AdExNeuron")


class TestARecordSaysWhetherItsSourcesAreStillTheSame:
    """A cited record carries one of three words about the sources it binds."""

    def test_unchanged_sources_make_the_record_current(self, tmp_path: Path) -> None:
        """Every bound digest equals its file, so nothing is listed as changed."""
        _tree_with_one_record(tmp_path)
        row = _adex_row(build_census(tmp_path))
        assert row["state"] == "measured"
        assert row["records"] == [{"name": "bench_adex.json", "currency": "current"}]

    def test_one_changed_source_makes_it_outdated_and_is_named(self, tmp_path: Path) -> None:
        """Only the source that changed is listed; the untouched one is not."""
        source, _record = _tree_with_one_record(tmp_path)
        source.write_text("STEP = 2\n", encoding="utf-8")
        row = _adex_row(build_census(tmp_path))
        assert row["state"] == "measured"
        assert row["records"] == [
            {"name": "bench_adex.json", "currency": "outdated", "changed": ["src/kernel.py"]}
        ]

    def test_a_removed_source_makes_it_outdated_and_is_named(self, tmp_path: Path) -> None:
        """A bound source that no longer exists is a change, not a pass."""
        source, _record = _tree_with_one_record(tmp_path)
        source.unlink()
        entry = record_currency("bench_adex.json", tmp_path)
        assert entry == {
            "name": "bench_adex.json",
            "currency": "outdated",
            "changed": ["src/kernel.py"],
        }

    @pytest.mark.parametrize(
        "payload",
        [
            {"model": "AdExNeuron"},
            {"model": "AdExNeuron", "source_hashes": {"python": "0" * 64}},
            {"model": "AdExNeuron", "source_hashes": {}},
        ],
    )
    def test_a_record_that_binds_no_path_is_unbound(
        self, tmp_path: Path, payload: dict[str, object]
    ) -> None:
        """Without a digest keyed by a path nothing can be said either way."""
        _source, record = _tree_with_one_record(tmp_path)
        record.write_text(json.dumps(payload), encoding="utf-8")
        row = _adex_row(build_census(tmp_path))
        assert row["state"] == "measured"
        assert row["records"] == [{"name": "bench_adex.json", "currency": "unbound"}]

    def test_a_record_in_a_subfolder_is_named_by_its_path(self, tmp_path: Path) -> None:
        """Two records with the same file name in different folders stay apart."""
        _tree_with_one_record(tmp_path, folder="nested")
        row = _adex_row(build_census(tmp_path))
        assert row["records"] == [{"name": "nested/bench_adex.json", "currency": "current"}]

    def test_the_record_totals_count_each_record_once(self, tmp_path: Path) -> None:
        """A record that names two models is one record."""
        source, record = _tree_with_one_record(tmp_path)
        payload = json.loads(record.read_text(encoding="utf-8"))
        payload["also"] = "LapicqueNeuron"
        record.write_text(json.dumps(payload), encoding="utf-8")
        source.write_text("STEP = 3\n", encoding="utf-8")
        census = build_census(tmp_path)
        assert census["cited_records"] == 1
        assert census["record_currency_totals"] == {"current": 0, "outdated": 1, "unbound": 0}
        totals = census["totals"]
        assert isinstance(totals, dict)
        assert totals["measured"] == 2


class TestTheCommittedCensusCarriesCurrency:
    """The real census states a currency word for every record it cites."""

    def test_every_cited_record_has_one_of_the_three_words(self, census: dict[str, object]) -> None:
        """An outdated record names what changed; the others list nothing."""
        seen: dict[str, str] = {}
        for row in _rows(census):
            records = row["records"]
            assert isinstance(records, list)
            for record in records:
                assert isinstance(record, dict)
                word = str(record["currency"])
                assert word in CURRENCIES
                assert seen.setdefault(str(record["name"]), word) == word
                if word == "outdated":
                    assert record["changed"] == sorted(set(record["changed"]))
                    assert record["changed"]
                else:
                    assert "changed" not in record
        record_totals = census["record_currency_totals"]
        assert isinstance(record_totals, dict)
        assert census["cited_records"] == len(seen)
        for word in CURRENCIES:
            assert record_totals[word] == sum(1 for value in seen.values() if value == word)

    def test_the_summary_says_what_outdated_means(self, census: dict[str, object]) -> None:
        """A reader of the summary must not take an outdated record for a slow one."""
        summary = render_summary(census)
        for word in CURRENCIES:
            assert word in summary
        assert "does not say the speed changed" in summary


def _uncited_record(root: Path, name: str = "local_two_compartment_rk4.json") -> tuple[Path, Path]:
    """Write a source and a record that binds it but names no catalogue model."""
    source = root / "src" / "lane.rs"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("fn step() {}\n", encoding="utf-8")
    record = root / "benchmarks" / "results" / name
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(
        json.dumps(
            {
                "benchmark": "two_compartment_rk4",
                "source_hashes": {"src/lane.rs": _digest(source)},
            }
        ),
        encoding="utf-8",
    )
    return source, record


class TestARecordNoModelNamesIsStillWatched:
    """A record that binds sources is watched even when no model's name is in it."""

    def test_it_is_listed_with_its_currency(self, tmp_path: Path) -> None:
        """No catalogue model is measured by it, and its sources are still followed."""
        _uncited_record(tmp_path)
        census = build_census(tmp_path)
        assert census["cited_records"] == 0
        assert census["uncited_bound_records"] == [
            {"name": "local_two_compartment_rk4.json", "currency": "current"}
        ]
        assert census["all_records"] == 1
        assert census["all_record_currency_totals"] == {"current": 1, "outdated": 0, "unbound": 0}

    def test_a_changed_source_is_named(self, tmp_path: Path) -> None:
        """The edit that no model row would show appears here."""
        source, _record = _uncited_record(tmp_path)
        source.write_text("fn step() { /* documented */ }\n", encoding="utf-8")
        census = build_census(tmp_path)
        assert census["uncited_bound_records"] == [
            {
                "name": "local_two_compartment_rk4.json",
                "currency": "outdated",
                "changed": ["src/lane.rs"],
            }
        ]
        assert census["all_record_currency_totals"] == {"current": 0, "outdated": 1, "unbound": 0}

    def test_one_that_binds_no_path_is_counted_and_not_listed(self, tmp_path: Path) -> None:
        """Nothing can be said about it, so it adds to the total only."""
        _source, record = _uncited_record(tmp_path)
        record.write_text(json.dumps({"benchmark": "two_compartment_rk4"}), encoding="utf-8")
        census = build_census(tmp_path)
        assert census["uncited_bound_records"] == []
        assert census["all_records"] == 1
        assert census["all_record_currency_totals"] == {"current": 0, "outdated": 0, "unbound": 1}

    def test_a_cited_record_is_not_listed_a_second_time(self, tmp_path: Path) -> None:
        """A record a model row carries is counted once in the totals over all."""
        _tree_with_one_record(tmp_path)
        _uncited_record(tmp_path, name="formal/receipt.json")
        census = build_census(tmp_path)
        assert census["cited_records"] == 1
        assert census["uncited_bound_records"] == [
            {"name": "formal/receipt.json", "currency": "current"}
        ]
        assert census["all_records"] == 2
        assert census["all_record_currency_totals"] == {"current": 2, "outdated": 0, "unbound": 0}

    def test_a_file_that_is_not_json_is_not_a_record(self, tmp_path: Path) -> None:
        """It is neither counted nor listed."""
        _uncited_record(tmp_path)
        (tmp_path / "benchmarks" / "results" / "broken.json").write_text("{", encoding="utf-8")
        census = build_census(tmp_path)
        assert census["all_records"] == 1

    def test_the_committed_census_watches_every_record_that_parses(
        self, census: dict[str, object]
    ) -> None:
        """Counted here by a separate walk of the results directory."""
        results = REPO_ROOT / "benchmarks" / "results"
        parsed = 0
        for path in results.rglob("*.json"):
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            parsed += 1
        assert census["all_records"] == parsed
        all_totals = census["all_record_currency_totals"]
        cited_totals = census["record_currency_totals"]
        assert isinstance(all_totals, dict)
        assert isinstance(cited_totals, dict)
        assert sum(int(all_totals[word]) for word in CURRENCIES) == parsed
        cited = {str(record["name"]) for row in _rows(census) for record in _records(row)}
        others = census["uncited_bound_records"]
        assert isinstance(others, list)
        names = [str(entry["name"]) for entry in others]
        assert names == sorted(set(names))
        assert cited.isdisjoint(names)
        for entry in others:
            assert entry["currency"] in ("current", "outdated")
            assert ("changed" in entry) == (entry["currency"] == "outdated")
        for word in ("current", "outdated"):
            listed = sum(1 for entry in others if entry["currency"] == word)
            assert int(all_totals[word]) == int(cited_totals[word]) + listed

    def test_the_summary_counts_them(self, tmp_path: Path) -> None:
        """An operator sees how many records exist beyond the ones a model cites."""
        _tree_with_one_record(tmp_path)
        _uncited_record(tmp_path)
        summary = render_summary(build_census(tmp_path))
        assert "all records: 2" in summary
        assert "bound to sources and named by no model: 1" in summary


def _lane_keyed_record(root: Path) -> tuple[Path, Path, Path]:
    """Write a source, a record that keys its digest by lane, and the gate manifest naming the path."""
    source = root / "src" / "lane.go"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("package lane\n", encoding="utf-8")
    record = root / "benchmarks" / "results" / "bench_lane_keyed.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"source_hashes": {"go": _digest(source)}}), encoding="utf-8")
    manifest = root / "benchmarks" / "benchmark_regression_gates.json"
    manifest.write_text(
        json.dumps(
            {
                "gates": [
                    {
                        "id": "lane-keyed",
                        "artefact": "benchmarks/results/bench_lane_keyed.json",
                        "source_hashes": {"src/lane.go": "source_hashes.go"},
                    },
                    {"id": "no-mapping", "artefact": "benchmarks/results/other.json"},
                    "not a gate",
                ]
            }
        ),
        encoding="utf-8",
    )
    return source, record, manifest


class TestTheGateManifestNamesThePathsOfLaneKeyedRecords:
    """A record that keys digests by lane is bound through the evidence gate's manifest."""

    def test_the_manifest_binds_it(self, tmp_path: Path) -> None:
        """Without the manifest nothing could be said; with it the record is current."""
        _lane_keyed_record(tmp_path)
        census = build_census(tmp_path)
        assert census["uncited_bound_records"] == [
            {"name": "bench_lane_keyed.json", "currency": "current"}
        ]

    def test_a_changed_source_is_named_by_its_path(self, tmp_path: Path) -> None:
        """The path comes from the manifest, not from the lane key."""
        source, _record, _manifest = _lane_keyed_record(tmp_path)
        source.write_text("package lane // edited\n", encoding="utf-8")
        assert build_census(tmp_path)["uncited_bound_records"] == [
            {"name": "bench_lane_keyed.json", "currency": "outdated", "changed": ["src/lane.go"]}
        ]

    def test_a_named_digest_the_record_lacks_counts_as_changed(self, tmp_path: Path) -> None:
        """A source the manifest names without a recorded digest cannot be called current."""
        _source, record, _manifest = _lane_keyed_record(tmp_path)
        record.write_text(json.dumps({"source_hashes": {"rust": "0" * 64}}), encoding="utf-8")
        assert build_census(tmp_path)["uncited_bound_records"] == [
            {"name": "bench_lane_keyed.json", "currency": "outdated", "changed": ["src/lane.go"]}
        ]

    @pytest.mark.parametrize("content", [None, "{", "[]", '{"gates": "none"}'])
    def test_without_a_usable_manifest_the_record_is_unbound(
        self, tmp_path: Path, content: str | None
    ) -> None:
        """An absent or malformed manifest binds nothing and breaks nothing."""
        _source, _record, manifest = _lane_keyed_record(tmp_path)
        if content is None:
            manifest.unlink()
        else:
            manifest.write_text(content, encoding="utf-8")
        census = build_census(tmp_path)
        assert census["uncited_bound_records"] == []
        assert census["all_record_currency_totals"] == {"current": 0, "outdated": 0, "unbound": 1}

    def test_digests_keyed_by_path_are_read_from_the_record_itself(self, tmp_path: Path) -> None:
        """The manifest is consulted only when the record names no path."""
        _source, record = _uncited_record(tmp_path)
        manifest = tmp_path / "benchmarks" / "benchmark_regression_gates.json"
        manifest.write_text(
            json.dumps(
                {
                    "gates": [
                        {
                            "artefact": f"benchmarks/results/{record.name}",
                            "source_hashes": {"src/absent.rs": "source_hashes.absent"},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        assert build_census(tmp_path)["uncited_bound_records"] == [
            {"name": record.name, "currency": "current"}
        ]


class TestUnreadableFilesAreSkippedNotCounted:
    """A file that cannot be read is neither evidence nor apparatus."""

    def test_a_record_that_is_not_json_names_nothing(self, tmp_path: Path) -> None:
        """A broken record beside a good one does not stop the count."""
        _tree_with_one_record(tmp_path)
        broken = tmp_path / "benchmarks" / "results" / "broken.json"
        broken.write_text('{"model": "LapicqueNeuron"', encoding="utf-8")
        assert measured_models(tmp_path) == {"AdExNeuron": ["bench_adex.json"]}

    def test_a_script_path_that_cannot_be_read_names_nothing(self, tmp_path: Path) -> None:
        """A folder whose name ends like a script is skipped."""
        scripts = tmp_path / "benchmarks"
        (scripts / "trap.py").mkdir(parents=True)
        (scripts / "bench_adex.py").write_text("MODEL = 'AdExNeuron'\n", encoding="utf-8")
        assert apparatus_models(tmp_path) == {"AdExNeuron": ["bench_adex.py"]}


class TestTheCommandLineIsTheDriftGate:
    """The check mode is what refuses a census that no longer matches the tree."""

    def test_a_written_census_passes_its_own_check(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Writing and then checking the same file succeeds."""
        output = tmp_path / "out" / "census.json"
        assert main(["--write", "--summary", "--output", str(output)]) == 0
        written = capsys.readouterr().out
        assert "Wrote" in written
        assert "catalogue models: 185" in written
        assert output.read_text(encoding="utf-8") == encode(build_census())
        assert main(["--check", "--output", str(output)]) == 0
        assert "is up to date" in capsys.readouterr().out

    def test_an_absent_census_fails_the_check(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A missing file is a failure that says what to run."""
        assert main(["--check", "--output", str(tmp_path / "missing.json")]) == 1
        assert "is absent; run --write" in capsys.readouterr().err

    def test_a_census_that_no_longer_matches_fails_the_check(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """One record reported with another word than the tree gives is refused."""
        output = tmp_path / "census.json"
        assert main(["--write", "--output", str(output)]) == 0
        capsys.readouterr()
        census = json.loads(output.read_text(encoding="utf-8"))
        assert "outdated" in output.read_text(encoding="utf-8")
        for row in census["rows"]:
            for record in row["records"]:
                if record["currency"] == "outdated":
                    record["currency"] = "current"
                    record.pop("changed")
        output.write_text(json.dumps(census, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        assert main(["--check", "--output", str(output)]) == 1
        assert "has drifted; run --write" in capsys.readouterr().err

    def test_no_argument_prints_the_summary(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Run bare, the tool reports and writes nothing."""
        assert main([]) == 0
        printed = capsys.readouterr().out
        assert "cited records:" in printed
        assert "Wrote" not in printed

    def test_the_script_runs_as_a_program(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Started as a program, the committed census passes its check."""
        script = REPO_ROOT / "tools" / "benchmark_coverage_census.py"
        monkeypatch.setattr(sys, "argv", [str(script), "--check"])
        with pytest.raises(SystemExit) as stopped:
            runpy.run_path(str(script), run_name="__main__")
        assert stopped.value.code == 0
        assert "is up to date" in capsys.readouterr().out
