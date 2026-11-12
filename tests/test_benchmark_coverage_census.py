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
a benchmark script, and 84 are measured by a *committed record* (28 by a record
that holds their class name, 56 more by a record that binds their benchmark
script). The 17 models between those figures own the apparatus and show no
evidence anyone ran it.

These cases hold the census to that distinction. A script is not a measurement,
the three states stay exhaustive and disjoint, and the totals are pinned so the
gap shrinks only by a deliberate test change — never by a scanner quietly
counting scripts as results again.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import runpy
import subprocess
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
PINNED_TOTALS = {"measured": 84, "apparatus_only": 17, "absent": 84}


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
        """A citation that does not name the model is not evidence for it.

        Checked here by reading the files: a record attributed by class name holds
        the name; one attributed through its script binds that script by path, and
        the script names this model and no other.
        """
        every_name = [str(other["model"]) for other in _rows(census)]
        for row in _rows(census):
            if row["state"] != "measured":
                continue
            records = _records(row)
            assert records
            for record in records:
                path = REPO_ROOT / "benchmarks" / "results" / str(record["name"])
                text = path.read_text(encoding="utf-8")
                if record["attribution"] == "class_name":
                    assert f'"{row["model"]}"' in text
                    assert "script" not in record
                    continue
                assert record["attribution"] == "bound_script"
                script = str(record["script"])
                assert f"benchmarks/{script}" in json.loads(text)["source_hashes"]
                source = (REPO_ROOT / "benchmarks" / script).read_text(encoding="utf-8")
                named = [name for name in every_name if re.search(rf"\b{name}\b", source)]
                assert named == [str(row["model"])]

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
        assert row["records"] == [
            {"name": "bench_adex.json", "currency": "current", "attribution": "class_name"}
        ]

    def test_one_changed_source_makes_it_outdated_and_is_named(self, tmp_path: Path) -> None:
        """Only the source that changed is listed; the untouched one is not."""
        source, _record = _tree_with_one_record(tmp_path)
        source.write_text("STEP = 2\n", encoding="utf-8")
        row = _adex_row(build_census(tmp_path))
        assert row["state"] == "measured"
        assert row["records"] == [
            {
                "name": "bench_adex.json",
                "currency": "outdated",
                "changed": ["src/kernel.py"],
                "attribution": "class_name",
            }
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
        assert row["records"] == [
            {"name": "bench_adex.json", "currency": "unbound", "attribution": "class_name"}
        ]

    def test_a_record_in_a_subfolder_is_named_by_its_path(self, tmp_path: Path) -> None:
        """Two records with the same file name in different folders stay apart."""
        _tree_with_one_record(tmp_path, folder="nested")
        row = _adex_row(build_census(tmp_path))
        assert row["records"] == [
            {"name": "nested/bench_adex.json", "currency": "current", "attribution": "class_name"}
        ]

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


def _add_untracked_and_ignored(root: Path) -> None:
    """Put a record and a script on disk that a repository would not track."""
    local = root / "benchmarks" / "results" / "experimental_runs"
    local.mkdir(parents=True)
    (local / "trial.json").write_text(json.dumps({"model": "LapicqueNeuron"}), encoding="utf-8")
    (root / "benchmarks" / "results" / "scratch.json").write_text("{}", encoding="utf-8")
    (root / "benchmarks" / "bench_scratch.py").write_text("LapicqueNeuron\n", encoding="utf-8")


class TestOnlyTrackedFilesAreRead:
    """In a repository the census reads what git tracks, not what lies in the directory."""

    def test_untracked_and_ignored_files_change_nothing(self, tmp_path: Path) -> None:
        """A local run left beside the committed records is not evidence."""
        _tree_with_one_record(tmp_path)
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
        (tmp_path / ".gitignore").write_text(
            "benchmarks/results/experimental_runs/\n", encoding="utf-8"
        )
        subprocess.run(
            ["git", "-C", str(tmp_path), "add", ".gitignore", "src", "benchmarks"],
            check=True,
            capture_output=True,
        )
        before = encode(build_census(tmp_path))
        _add_untracked_and_ignored(tmp_path)
        after = build_census(tmp_path)
        assert encode(after) == before
        assert after["all_records"] == 1
        totals = after["totals"]
        assert isinstance(totals, dict)
        assert totals["measured"] == 1
        assert totals["apparatus_only"] == 0

    def test_without_a_repository_the_directory_is_walked(self, tmp_path: Path) -> None:
        """The same files in a plain tree are all read: the test above can tell the difference."""
        _tree_with_one_record(tmp_path)
        _add_untracked_and_ignored(tmp_path)
        census = build_census(tmp_path)
        assert census["all_records"] == 3
        totals = census["totals"]
        assert isinstance(totals, dict)
        assert totals["measured"] == 2

    def test_a_newly_staged_record_is_read(self, tmp_path: Path) -> None:
        """A record added to the index counts before it is committed."""
        _tree_with_one_record(tmp_path)
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(tmp_path), "add", "src", "benchmarks"],
            check=True,
            capture_output=True,
        )
        _uncited_record(tmp_path)
        assert build_census(tmp_path)["all_records"] == 1
        subprocess.run(
            ["git", "-C", str(tmp_path), "add", "src", "benchmarks"],
            check=True,
            capture_output=True,
        )
        assert build_census(tmp_path)["all_records"] == 2


def _script_bound_record(
    root: Path, script_text: str, *, script: str = "benchmarks/bench_demo.py"
) -> tuple[Path, Path]:
    """Write a benchmark script and a record that binds it by path and names no model."""
    script_path = root / script
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script_text, encoding="utf-8")
    record = root / "benchmarks" / "results" / "local_demo.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(
        json.dumps({"benchmark": "demo", "source_hashes": {script: _digest(script_path)}}),
        encoding="utf-8",
    )
    return script_path, record


def _row(census: dict[str, object], model: str) -> dict[str, object]:
    return next(row for row in _rows(census) if row["model"] == model)


class TestARecordIsAttributedThroughTheScriptItBinds:
    """A record that binds one benchmark script measures the one model that script names."""

    def test_the_model_of_the_bound_script_is_measured(self, tmp_path: Path) -> None:
        """The record says which script produced it; the script says which model."""
        _script_bound_record(tmp_path, "from models import AdExNeuron\n")
        census = build_census(tmp_path)
        row = _row(census, "AdExNeuron")
        assert row["state"] == "measured"
        assert row["records"] == [
            {
                "name": "local_demo.json",
                "currency": "current",
                "attribution": "bound_script",
                "script": "bench_demo.py",
            }
        ]
        assert census["cited_records"] == 1
        assert census["uncited_bound_records"] == []

    def test_a_script_that_names_two_models_attributes_nothing(self, tmp_path: Path) -> None:
        """Which of the two the record measured is not stated, so neither is credited."""
        _script_bound_record(tmp_path, "AdExNeuron\nLapicqueNeuron\n")
        census = build_census(tmp_path)
        assert _row(census, "AdExNeuron")["state"] == "apparatus_only"
        assert _row(census, "LapicqueNeuron")["state"] == "apparatus_only"
        assert census["cited_records"] == 0
        assert census["uncited_bound_records"] == [
            {"name": "local_demo.json", "currency": "current"}
        ]

    def test_a_name_inside_a_longer_name_is_not_a_mention(self, tmp_path: Path) -> None:
        """`NMDANeuron` is not named by a script that names `DendriticNMDANeuron`."""
        _script_bound_record(tmp_path, "neuron = DendriticNMDANeuron()\n")
        census = build_census(tmp_path)
        assert _row(census, "DendriticNMDANeuron")["state"] == "measured"
        shorter = _row(census, "NMDANeuron")
        assert shorter["state"] == "absent"
        assert shorter["scripts"] == []

    def test_a_name_inside_another_identifier_is_not_a_mention(self, tmp_path: Path) -> None:
        """`AdExNeuron` is not named by `NotAdExNeuronAtAll` or by `AdExNeuronTable`."""
        _script_bound_record(tmp_path, "class NotAdExNeuronAtAll:\n    AdExNeuronTable = 1\n")
        census = build_census(tmp_path)
        row = _row(census, "AdExNeuron")
        assert row["state"] == "absent"
        assert row["scripts"] == []
        assert census["cited_records"] == 0

    def test_a_class_name_in_the_record_comes_first(self, tmp_path: Path) -> None:
        """A record that names its model is not also credited to its script's model."""
        _script_path, record = _script_bound_record(tmp_path, "AdExNeuron\n")
        payload = json.loads(record.read_text(encoding="utf-8"))
        payload["model"] = "LapicqueNeuron"
        record.write_text(json.dumps(payload), encoding="utf-8")
        census = build_census(tmp_path)
        assert _row(census, "LapicqueNeuron")["records"] == [
            {"name": "local_demo.json", "currency": "current", "attribution": "class_name"}
        ]
        assert _row(census, "AdExNeuron")["state"] == "apparatus_only"
        assert census["cited_records"] == 1

    @pytest.mark.parametrize("script", ["benchmarks/helpers/bench_demo.py", "tools/bench_demo.py"])
    def test_only_a_script_of_the_benchmark_directory_counts(
        self, tmp_path: Path, script: str
    ) -> None:
        """A bound source elsewhere is a source, not the benchmark that wrote the record."""
        _script_bound_record(tmp_path, "AdExNeuron\n", script=script)
        census = build_census(tmp_path)
        assert _row(census, "AdExNeuron")["state"] == "absent"
        assert census["cited_records"] == 0

    def test_a_changed_script_keeps_the_attribution_and_ages_the_record(
        self, tmp_path: Path
    ) -> None:
        """Whom a record measures and whether it is current are two separate facts."""
        script_path, _record = _script_bound_record(tmp_path, "AdExNeuron\n")
        script_path.write_text("AdExNeuron  # edited\n", encoding="utf-8")
        assert _row(build_census(tmp_path), "AdExNeuron")["records"] == [
            {
                "name": "local_demo.json",
                "currency": "outdated",
                "changed": ["benchmarks/bench_demo.py"],
                "attribution": "bound_script",
                "script": "bench_demo.py",
            }
        ]

    def test_two_bound_scripts_of_one_model_are_one_attribution(self, tmp_path: Path) -> None:
        """Both scripts name the same single model; the first by name is recorded."""
        _first, record = _script_bound_record(tmp_path, "AdExNeuron\n")
        second = tmp_path / "benchmarks" / "bench_again.py"
        second.write_text("AdExNeuron\n", encoding="utf-8")
        payload = json.loads(record.read_text(encoding="utf-8"))
        payload["source_hashes"]["benchmarks/bench_again.py"] = _digest(second)
        record.write_text(json.dumps(payload), encoding="utf-8")
        records = _row(build_census(tmp_path), "AdExNeuron")["records"]
        assert isinstance(records, list)
        assert len(records) == 1
        assert records[0]["script"] == "bench_again.py"

    def test_a_bound_helper_that_names_no_model_is_not_the_script(self, tmp_path: Path) -> None:
        """A shared helper beside the benchmark is bound too; it attributes nothing."""
        _demo, record = _script_bound_record(tmp_path, "AdExNeuron\n")
        helper = tmp_path / "benchmarks" / "a_shared_context.py"
        helper.write_text("def load_average() -> str:\n    return ''\n", encoding="utf-8")
        payload = json.loads(record.read_text(encoding="utf-8"))
        payload["source_hashes"]["benchmarks/a_shared_context.py"] = _digest(helper)
        record.write_text(json.dumps(payload), encoding="utf-8")
        records = _row(build_census(tmp_path), "AdExNeuron")["records"]
        assert isinstance(records, list)
        assert [entry["script"] for entry in records] == ["bench_demo.py"]

    def test_two_bound_scripts_of_different_models_attribute_nothing(self, tmp_path: Path) -> None:
        """A record bound to the benchmarks of two models is credited to neither."""
        _first, record = _script_bound_record(tmp_path, "AdExNeuron\n")
        second = tmp_path / "benchmarks" / "bench_other.py"
        second.write_text("LapicqueNeuron\n", encoding="utf-8")
        payload = json.loads(record.read_text(encoding="utf-8"))
        payload["source_hashes"]["benchmarks/bench_other.py"] = _digest(second)
        record.write_text(json.dumps(payload), encoding="utf-8")
        census = build_census(tmp_path)
        assert census["cited_records"] == 0
        assert _row(census, "AdExNeuron")["state"] == "apparatus_only"

    def test_the_dendritic_nmda_record_measures_its_model(self, census: dict[str, object]) -> None:
        """The record that started this: it names its benchmark, not its class."""
        row = _row(census, "DendriticNMDANeuron")
        assert row["state"] == "measured"
        names = {str(record["name"]): record for record in _records(row)}
        record = names["local_python_2026-06-26_dendritic_nmda_rk4.json"]
        assert record["attribution"] == "bound_script"
        assert record["script"] == "bench_model_dendritic_nmda.py"

    def test_the_summary_says_how_the_measured_models_are_known(
        self, census: dict[str, object]
    ) -> None:
        """Two counts that add up to the cited records."""
        by = census["attribution_totals"]
        assert isinstance(by, dict)
        assert set(by) == {"class_name", "bound_script"}
        assert int(by["class_name"]) + int(by["bound_script"]) == census["cited_records"]
        summary = render_summary(census)
        assert f"by class name {by['class_name']}" in summary
        assert f"through the bound script {by['bound_script']}" in summary


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
