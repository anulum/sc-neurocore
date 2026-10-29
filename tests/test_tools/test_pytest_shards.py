# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Duration scheduler CLI and measured JUnit history

"""Exercise module assignment and history updates through the scheduler CLI."""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import subprocess

import pytest

from tools.pytest_shards import main, partition, read_timings


def test_weighted_assignment_balances_load_and_includes_new_modules() -> None:
    """Historical rows cannot omit new tests or add removed tests to discovery."""
    files = ["a", "b", "c", "new"]
    weights = {"a": 12.0, "b": 8.0, "c": 4.0, "removed": 4.0}
    shards = partition(files, 2, weights)
    assert shards == [["a", "c"], ["b", "new"]]
    assert sorted(sum(shards, [])) == files
    assert partition(files, 2, {}) == [["a", "c"], ["b", "new"]]


@pytest.mark.parametrize(("files", "count"), [(["a"], 2), (["a", "a"], 1), (["a"], 0), ([""], 1)])
def test_invalid_partition_refuses_missing_or_duplicate_members(
    files: list[str], count: int
) -> None:
    """Every shard must be nonempty and every discovered module unique."""
    with pytest.raises(ValueError, match="unique discovered"):
        partition(files, count, {})


@pytest.mark.parametrize(
    "payload",
    [
        "[]",
        '{"schema":2}',
        '{"schema":1,"durations":[]}',
        '{"schema":1,"durations":{"x":true}}',
        '{"schema":1,"durations":{"x":null}}',
        '{"schema":1,"durations":{"x":NaN}}',
        '{"schema":1,"durations":{"x":0}}',
    ],
)
def test_invalid_history_is_refused(tmp_path: Path, payload: str) -> None:
    """Malformed caches cannot change scheduling silently."""
    history = tmp_path / "history.json"
    history.write_text(payload)
    assert main(["select", "--shard=1/1", f"--timings={history}"]) == 2


def test_cold_history_and_valid_measurements(tmp_path: Path) -> None:
    """Missing hints preserve complete cold scheduling; valid integers become seconds."""
    history = tmp_path / "history.json"
    assert read_timings(None) == read_timings(history) == {}
    history.write_text('{"schema":1,"durations":{"a":2,"b":0.5}}')
    assert read_timings(history) == {"a": 2.0, "b": 0.5}


def test_scheduler_process_partitions_the_live_repository(tmp_path: Path) -> None:
    """All eight CLI selections cover actual discovered modules exactly once."""
    root = Path(__file__).resolve().parents[2]
    files = sorted(
        path.relative_to(root).as_posix()
        for path in (root / "tests").rglob("*.py")
        if path.name.startswith("test_") or path.name.endswith("_test.py")
    )
    timings = tmp_path / "durations.json"
    timings.write_text(json.dumps({"schema": 1, "durations": {files[0]: 12}}))
    selected = []
    for index in range(1, 9):
        result = subprocess.run(
            [
                sys.executable,
                str(root / "tools/pytest_shards.py"),
                "select",
                f"--shard={index}/8",
                f"--timings={timings}",
            ],
            input="\0".join(files) + "\0",
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        batch = result.stdout.rstrip("\0").split("\0")
        assert batch
        selected.extend(batch)
    assert sorted(selected) == files


@pytest.mark.parametrize("data", [b"alpha", b"alpha\0"])
def test_select_cli_preserves_nul_paths(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], data: bytes
) -> None:
    """The CLI accepts only complete NUL-terminated discovery and emits exact paths."""
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(data)))
    result = main(["select", "--shard=1/1"])
    captured = capsys.readouterr()
    assert result == (0 if data.endswith(b"\0") else 2)
    assert captured.out == ("alpha\0" if result == 0 else "")


@pytest.mark.parametrize("shard", ["0/2", "3/2", "bad"])
def test_select_cli_rejects_invalid_shards(shard: str) -> None:
    """Invalid addressing refuses before reading discovery."""
    assert main(["select", f"--shard={shard}"]) == 2


def test_record_cli_retains_unobserved_live_history_and_aggregates_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """JUnit setup/call/teardown totals update live modules without erasing partial history."""
    monkeypatch.chdir(tmp_path)
    tests = tmp_path / "tests"
    tests.mkdir()
    for name in ("test_first.py", "test_second.py", "third_test.py"):
        (tests / name).touch()
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "test-results-batch-001.xml").write_text(
        '<testsuite><testcase classname="tests.test_first.Class" time="1.5"/>'
        '<testcase classname="tests.test_first" time="2"/>'
        '<testcase classname="tests.third_test"/>'
        '<testcase classname="external.unknown" time="1"/></testsuite>'
    )
    output = tmp_path / "next" / "durations.json"
    previous = tmp_path / "previous.json"
    previous.write_text(
        json.dumps(
            {
                "schema": 1,
                "durations": {
                    "tests/test_second.py": 9,
                    "tests/removed.py": 5,
                },
            }
        )
    )
    assert (
        main(["record", f"--reports={reports}", f"--previous={previous}", f"--output={output}"])
        == 0
    )
    assert read_timings(output) == {
        "tests/test_first.py": 3.5,
        "tests/test_second.py": 9,
        "tests/third_test.py": 0.001,
    }


@pytest.mark.parametrize(
    "body",
    [
        "<broken>",
        "<testsuite/>",
        '<testsuite><testcase classname="tests.test_first" time="-1"/></testsuite>',
        '<testsuite><testcase classname="tests.test_first" time="NaN"/></testsuite>',
        # A downloaded report may not declare entities; the standard parser would expand this one.
        '<!DOCTYPE t [<!ENTITY m "tests.test_first">]>'
        '<testsuite><testcase classname="&m;" time="1"/></testsuite>',
    ],
)
def test_record_cli_refuses_invalid_reports_without_replacing_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str
) -> None:
    """An unreadable or empty observation cannot replace previous scheduling evidence."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_first.py").touch()
    (tmp_path / "test-results-batch-001.xml").write_text(body)
    output = tmp_path / "history.json"
    output.write_text("preserve")
    assert main(["record", f"--reports={tmp_path}", f"--output={output}"]) == 2
    assert output.read_text() == "preserve"
