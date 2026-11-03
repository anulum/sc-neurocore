# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real process-record input and snapshot-membership refusals

"""Damage copies of actual process metadata without fabricating a positive process or provider."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import cast

import pytest

from tools.go_coverage_environment import environment_mapping
from tools.go_coverage_records import (
    ProcessScope,
    Record,
    records,
    snapshot_directories,
    validate_process_header,
    validate_snapshot_membership,
)


def actual_scope(job: Path, phase: str) -> ProcessScope:
    """Keep the original real declarations when checking an explicitly damaged record copy."""
    return ProcessScope(
        job,
        phase,
        environment_mapping(json.loads((job / "python.json").read_text())),
        environment_mapping(json.loads((job / "environments" / (phase + ".json")).read_text())),
    )


def actual_before_rows(job: Path) -> list[Record]:
    """Read the completed actual before-work process instead of inventing an observer state."""
    path = next((job / "process-counters/before").glob("*/observations.jsonl"))
    return records(path)


@pytest.mark.parametrize("fault", ["duplicate-pid", "interpreter", "keys", "digest", "terminal"])
def test_corrupted_actual_header_inputs_cannot_change_process_custody(
    native_job: Path, fault: str
) -> None:
    """Real unchanged declarations refuse each separately damaged identity or environment field."""
    rows = deepcopy(actual_before_rows(native_job))
    seen: set[int] = set()
    if fault == "duplicate-pid":
        seen.add(cast(int, rows[0]["pid"]))
    elif fault == "interpreter":
        interpreter = environment_mapping(rows[0]["interpreter"])
        interpreter["prefix"] += "/different-prefix"
    else:
        closed = next(row for row in rows if row["event"] == "closed_environment")
        if fault == "keys":
            cast(list[str], closed["keys"]).pop()
        elif fault == "digest":
            closed["environment_sha256"] = "0" * 64
        else:
            terminal = environment_mapping(closed["terminal_environment"])
            terminal["COLUMNS"] = str(int(terminal["COLUMNS"]) + 1)
    with pytest.raises(RuntimeError, match="custody differs|metadata differ"):
        validate_process_header(actual_scope(native_job, "before"), rows, seen)


@pytest.mark.parametrize("fault", ["parent-type", "argv-type"])
def test_damaged_real_child_geometry_refuses_before_parent_declaration_lookup(
    native_job: Path, fault: str
) -> None:
    """A copied genuine child cannot turn a Boolean parent or non-string argument into identity."""
    children = [
        records(path)
        for path in (native_job / "process-counters/tests").glob("*/observations.jsonl")
    ]
    rows = deepcopy(
        next(row for row in children if cast(list[object], row[0]["argv"])[:1] == ["-c"])
    )
    if fault == "parent-type":
        rows[0]["ppid"] = True
    else:
        argv = cast(list[object], rows[0]["argv"])
        argv[1] = 0
    with pytest.raises(RuntimeError, match="parent or argv geometry"):
        validate_process_header(actual_scope(native_job, "tests"), rows, set())


@pytest.mark.parametrize("fault", ["empty", "duplicate", "role", "extra-process"])
def test_damaged_actual_snapshot_set_cannot_satisfy_before_work_membership(
    native_job: Path, fault: str
) -> None:
    """Mutate only copied metadata and paths, preserving every original native byte and stream."""
    rows = deepcopy(actual_before_rows(native_job))
    processes = {cast(int, rows[0]["pid"]): rows}
    snapshots = list(snapshot_directories(native_job, "before"))
    if fault == "empty":
        snapshots.clear()
    elif fault == "duplicate":
        snapshots.append(snapshots[0])
    elif fault == "role":
        next(row for row in rows if row["event"] == "pre_dlopen")["backend"] = "mojo"
    else:
        collector = json.loads((native_job / "collection-process.json").read_text())["pid"]
        assert collector not in processes
        path = native_job / "process-counters/tests" / str(collector) / "observations.jsonl"
        processes[collector] = records(path)
    with pytest.raises(RuntimeError, match="snapshot directories|preload membership"):
        validate_snapshot_membership(native_job, "before", processes, snapshots)
