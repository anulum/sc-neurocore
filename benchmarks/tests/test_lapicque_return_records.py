# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Validate genuine Lapicque return observations through their recording API

"""Replay real return observations and reject explicitly damaged copies of their contracts."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from types import CodeType, FrameType

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import capture_lapicque_samples as capture

Event = tuple[CodeType, int, object]
Observation = tuple[capture.TimingFrame, Event]


@pytest.fixture(scope="module")
def original_returns(
    pytestconfig: pytest.Config, tmp_path_factory: pytest.TempPathFactory
) -> list[Observation]:
    """Run all original work once and retain real interpreter frames and result objects."""
    root = Path(pytestconfig.rootpath)
    harness = str((root / "benchmarks/bench_model_lapicque.py").resolve())
    returned: list[tuple[FrameType, Event]] = []
    previous_profile = sys.getprofile()
    previous_affinity = os.sched_getaffinity(0)

    def observe(frame: FrameType, event: str, result: object) -> None:
        """Retain only actual returns from the original measured function."""
        if (
            frame.f_code.co_filename == harness
            and frame.f_code.co_name == "_measure_backend"
            and event == "return"
        ):
            returned.append((frame, (frame.f_code, frame.f_lasti, result)))

    output = tmp_path_factory.mktemp("actual-original-returns") / "complete-diagnostic"
    try:
        os.sched_setaffinity(0, {min(previous_affinity)})
        sys.setprofile(observe)
        assert (
            capture.main(["--checkout", str(root), "--output-dir", str(output), "--diagnostic"])
            == 0
        )
    finally:
        sys.setprofile(previous_profile)
        os.sched_setaffinity(0, previous_affinity)
    record = json.loads((output / "original-seven-duration-samples.json").read_text())
    assert record["raw_retention_valid"] is True
    assert record["performance_admissible"] is False
    assert len(returned) == len(record["rows"]) == 5
    snapshots = [(capture.TimingFrame.capture(frame), event) for frame, event in returned]
    assert [snapshot.values["backend"] for snapshot, _ in snapshots] == [
        "python",
        "rust",
        "julia",
        "go",
        "mojo",
    ]
    for (snapshot, event), row in zip(snapshots, record["rows"], strict=True):
        assert snapshot.code is event[0]
        assert snapshot.values["elapsed_ms"] == row["samples_ms"]
    return snapshots


@pytest.mark.parametrize("index", range(5), ids=["python", "rust", "julia", "go", "mojo"])
def test_real_return_replay_matches_original_samples(
    original_returns: list[Observation], index: int
) -> None:
    """Every real lane reproduces its original retained samples and aggregates."""
    frame, event = original_returns[index]
    records = capture.TimingReturns(event[0], diagnostic=True)
    capture.record_return(records, frame, event)
    assert records.errors == []
    assert len(records.rows) == 1
    assert records.rows[0]["samples_ms"] == frame.values["elapsed_ms"]
    result = event[2]
    assert isinstance(result, tuple)
    assert records.rows[0]["median_call_ms"] == result[0]
    assert records.rows[0]["minimum_call_ms"] == result[1]
    assert records.rows[0]["instruction_offset"] == event[1]


@pytest.mark.parametrize(
    "damage",
    [
        "event-code",
        "frame-code",
        "backend-absent",
        "backend-type",
        "backend-name",
        "samples-absent",
        "samples-type",
        "sample-count",
        "sample-bool",
        "sample-text",
        "sample-nan",
        "sample-infinite",
        "sample-zero",
        "sample-negative",
        "result-type",
        "result-size",
        "median",
        "minimum",
    ],
)
def test_damaged_copies_of_real_returns_are_rejected(
    original_returns: list[Observation], damage: str
) -> None:
    """Explicit negative copies never replace successful measurements or add valid rows."""
    frame, event = original_returns[0]
    values = dict(frame.values)
    code, offset, result = event
    assert isinstance(result, tuple)
    expected = ""
    if damage == "event-code":
        code = capture.main.__code__
        expected = "return frame does not match"
    elif damage == "frame-code":
        frame = replace(frame, code=capture.main.__code__)
        expected = "return frame does not match"
    elif damage.startswith("backend-"):
        if damage == "backend-absent":
            values.pop("backend")
        else:
            values["backend"] = False if damage == "backend-type" else "unrecognised-backend"
        expected = "unknown measured backend"
    elif damage in {"samples-absent", "samples-type", "sample-count"}:
        samples = values["elapsed_ms"]
        assert isinstance(samples, list)
        values["elapsed_ms"] = (
            None
            if damage == "samples-absent"
            else tuple(samples)
            if damage == "samples-type"
            else samples[:-1]
        )
        expected = "original seven timing samples are missing"
    elif damage.startswith("sample-"):
        samples = values["elapsed_ms"]
        assert isinstance(samples, list)
        changed = list(samples)
        invalid: dict[str, object] = {
            "sample-bool": False,
            "sample-text": "damaged",
            "sample-nan": float("nan"),
            "sample-infinite": float("inf"),
            "sample-zero": 0.0,
            "sample-negative": -1.0,
        }
        changed[0] = invalid[damage]
        values["elapsed_ms"] = changed
        expected = "invalid original elapsed sample"
    elif damage in {"result-type", "result-size"}:
        result = list(result) if damage == "result-type" else result[:-1]
        expected = "original result packet changed"
    else:
        changed_result = list(result)
        selected = 0 if damage == "median" else 1
        original = changed_result[selected]
        assert isinstance(original, float)
        changed_result[selected] = original + 1.0
        result = tuple(changed_result)
        expected = "raw samples disagree"
    damaged = replace(frame, values=values)
    records = capture.TimingReturns(event[0], diagnostic=True)
    capture.record_return(records, damaged, (code, offset, result))
    assert records.rows == []
    assert len(records.errors) == 1 and expected in records.errors[0]
    assert original_returns[0][0].values["backend"] == "python"
    assert original_returns[0][0].values["elapsed_ms"] == frame.values["elapsed_ms"]


def test_duplicate_genuine_return_keeps_only_its_first_record(
    original_returns: list[Observation],
) -> None:
    """Repeated delivery of an actual event records the duplication without adding data."""
    frame, event = original_returns[0]
    records = capture.TimingReturns(event[0], diagnostic=True)
    capture.record_return(records, frame, event)
    capture.record_return(records, frame, event)
    assert len(records.rows) == 1
    assert len(records.errors) == 1 and "duplicate measured backend" in records.errors[0]


def test_real_foreign_profiler_rejects_default_return_replay(
    original_returns: list[Observation],
) -> None:
    """Default recording refuses actual profiling rather than inventing an instrumentation flag."""
    frame, event = original_returns[0]
    previous = sys.getprofile()

    def observe(frame: FrameType, event: str, result: object) -> None:
        """Own a real profiling callback for the duration of the admission check."""

    try:
        sys.setprofile(observe)
        records = capture.TimingReturns(event[0], diagnostic=False)
        capture.record_return(records, frame, event)
        assert records.rows == []
        assert len(records.errors) == 1 and "foreign instrumentation" in records.errors[0]
        assert sys.getprofile() is observe
    finally:
        sys.setprofile(previous)


def test_complete_real_return_sequence_preserves_backend_order(
    original_returns: list[Observation],
) -> None:
    """One collection admits all five distinct genuine returns in original order."""
    records = capture.TimingReturns(original_returns[0][1][0], diagnostic=True)
    for frame, event in original_returns:
        capture.record_return(records, frame, event)
    assert records.errors == []
    assert [row["backend"] for row in records.rows] == ["python", "rust", "julia", "go", "mojo"]
