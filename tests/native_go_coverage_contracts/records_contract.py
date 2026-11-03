# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual native process and snapshot conservation contracts

"""Refuse damaged copies of actual current observer streams and native artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import cast

import pytest

from tools.go_coverage_images import read_images
from tools.go_coverage_records import (
    process_snapshots,
    records,
    snapshot_directories,
    verify_abi_processes,
)


def actual_parent(job: Path) -> Path:
    """Locate the actual complete ABI parent from its retained execution PID."""
    pid = json.loads((job / "execution.json").read_text())["pid"]
    return job / "process-counters/tests" / str(pid) / "observations.jsonl"


def test_actual_native_streams_conserve_current_loads_and_snapshots(native_job: Path) -> None:
    """The original completed parent stream still binds every real native snapshot."""
    path = actual_parent(native_job)
    paths = process_snapshots(
        path.parent, records(path), read_images(native_job / "images.json", native_job)
    )
    assert len(paths) == 4 and len(set(paths)) == 4
    assert len(snapshot_directories(native_job, "before")) == 4
    with pytest.raises(ValueError, match="Only complete"):
        snapshot_directories(native_job, "unknown")


@pytest.mark.parametrize(
    "fault", ["empty", "nonmapping", "terminal", "duplicate", "pid", "pid-type", "event"]
)
def test_actual_stream_damage_is_not_a_completed_process(
    native_job: Path,
    tmp_path: Path,
    fault: str,
) -> None:
    """A modified copy cannot manufacture registration, PID or successful completion."""
    original = actual_parent(native_job)
    rows = [json.loads(line) for line in original.read_text().splitlines()]
    if fault == "empty":
        rows = []
    elif fault == "nonmapping":
        rows[0] = "not a record"
    elif fault == "terminal":
        rows.pop()
    elif fault == "duplicate":
        rows.append(dict(rows[-1]))
    elif fault == "pid":
        rows[-1]["pid"] += 1
    elif fault == "pid-type":
        rows[0]["pid"] = True
    else:
        rows.insert(1, {"event": "invented event"})
    destination = tmp_path / original.parent.name
    destination.mkdir()
    path = destination / "observations.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises((ValueError, RuntimeError)):
        records(path)


@pytest.mark.parametrize(
    "fault",
    [
        "load",
        "load-digest",
        "refusal-missing",
        "refusal-status",
        "snapshot-missing",
        "capture",
        "snapshot-duplicate",
        "snapshot-status",
        "snapshot-digest",
        "destination",
        "files",
        "maps",
        "mojo",
    ],
)
def test_damaged_actual_load_or_snapshot_conservation_refuses(
    native_job: Path,
    fault: str,
) -> None:
    """No altered record can hide a missing or foreign native consumption artifact."""
    path = actual_parent(native_job)
    rows = records(path)
    load = next(row for row in rows if row.get("event") == "pre_dlopen")
    refusal = next(row for row in rows if row.get("event") == "invalid_snapshot_path")
    snapshot = next(row for row in rows if row.get("event") == "snapshot")
    if fault == "load":
        load["path"] = str(native_job / "absent.so")
    elif fault == "load-digest":
        load["sha256"] = "0" * 64
    elif fault == "refusal-missing":
        rows.remove(refusal)
    elif fault == "refusal-status":
        refusal["status"] = 0
    elif fault == "snapshot-missing":
        rows.remove(snapshot)
    elif fault == "capture":
        snapshot["capture"] = "invented"
    elif fault == "snapshot-duplicate":
        rows.append(dict(snapshot))
    elif fault == "snapshot-status":
        snapshot["status"] = 1
    elif fault == "snapshot-digest":
        snapshot["sha256"] = "0" * 64
    elif fault == "destination":
        snapshot["destination"] = str(native_job)
    elif fault == "files":
        snapshot["files"] = []
    elif fault == "maps":
        snapshot["maps"] = []
    else:
        rows.remove(next(row for row in rows if row.get("event") == "retained_mojo_mapping"))
    with pytest.raises(RuntimeError):
        process_snapshots(path.parent, rows, read_images(native_job / "images.json", native_job))


def test_actual_missing_counter_copy_refuses_without_touching_originals(
    native_job: Path,
    tmp_path: Path,
) -> None:
    """Removing actual copied counter bytes cannot pass snapshot artifact admission."""
    path = actual_parent(native_job)
    rows = records(path)
    process = tmp_path / path.parent.name
    process.mkdir()
    original_digests: dict[Path, bytes] = {}
    for row in rows:
        if row.get("event") == "snapshot":
            original = Path(str(row["destination"]))
            destination = process / original.parent.name / original.name
            shutil.copytree(original, destination)
            original_digests.update({p: p.read_bytes() for p in original.iterdir()})
            row["destination"] = str(destination)
    images = read_images(native_job / "images.json", native_job)
    snapshots = process_snapshots(process, rows, images)
    next(snapshots[0].glob("covcounters.*")).unlink()
    with pytest.raises(RuntimeError, match="missing"):
        process_snapshots(process, rows, images)
    assert all(path.read_bytes() == data for path, data in original_digests.items())


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "child-missing",
        "parent",
        "argv",
        "ppid",
        "role",
        "collector-launch",
        "child-geometry",
        "child-image",
        "failed-test",
    ],
)
def test_actual_abi_child_cohort_cannot_be_replaced_by_declarations(
    native_job: Path,
    tmp_path: Path,
    fault: str,
) -> None:
    """Full real launch/registration membership survives only its unchanged exact cohort."""
    processes = {
        int(path.parent.name): records(path)
        for path in (native_job / "process-counters/tests").glob("*/observations.jsonl")
    }
    execution = json.loads((native_job / "execution.json").read_text())
    collector = json.loads((native_job / "collection-process.json").read_text())
    parent = processes[execution["pid"]]
    child = next(
        rows for pid, rows in processes.items() if pid not in {execution["pid"], collector["pid"]}
    )
    if fault == "child-missing":
        del processes[cast(int, child[0]["pid"])]
    elif fault == "parent":
        execution["pid"] = collector["pid"]
    elif fault == "argv":
        child[0]["argv"] = ["-c", "changed"]
    elif fault == "ppid":
        child[0]["ppid"] = -1
    elif fault == "role":
        next(row for row in parent if row.get("event") == "pre_dlopen")["backend"] = "foreign"
    elif fault == "collector-launch":
        launch = next(row for row in parent if row.get("event") == "abi_child_launch")
        processes[collector["pid"]].append(dict(launch))
    elif fault == "child-geometry":
        launch = next(
            row
            for row in parent
            if row.get("event") == "abi_child_launch"
            and row["expected_child_argv"] == child[0]["argv"]
        )
        child[0]["argv"] = ["-c", "damaged-actual-geometry"]
        launch["expected_child_argv"] = child[0]["argv"]
    elif fault == "child-image":
        next(row for row in child if row.get("event") == "pre_dlopen")["model"] = "foreign-model"
    elif fault == "failed-test":
        execution["pytest_exit"] = 1
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "execution.json").write_text(json.dumps(execution))
    (tmp_path / "collection-process.json").write_text(json.dumps(collector))
    if fault == "none":
        verify_abi_processes(tmp_path, processes)
        actual = json.loads((tmp_path / "artifacts/abi-process-membership.json").read_text())
        assert actual["actual_original_children"] == actual["matched_launches"] == 72
    else:
        with pytest.raises(RuntimeError):
            verify_abi_processes(tmp_path, processes)


@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_actual_closed_terminal_record_damage_refuses_before_snapshot_consumption(
    native_job: Path, tmp_path: Path, damage: str
) -> None:
    """An altered terminal pair in a real before-work stream cannot pass reconciliation."""
    view = tmp_path / "terminal-record-view"
    (view / "environments").mkdir(parents=True)
    shutil.copyfile(native_job / "environments/before.json", view / "environments/before.json")
    shutil.copyfile(native_job / "python.json", view / "python.json")
    images = json.loads((native_job / "images.json").read_text())
    for row in images:
        original = Path(row["path"])
        copied = view / "images" / row["backend"] / row["model"] / original.name
        copied.parent.mkdir(parents=True)
        shutil.copyfile(original, copied)
        row["path"] = str(copied)
    (view / "images.json").write_text(json.dumps(images))
    original_stream = next((native_job / "process-counters/before").glob("*/observations.jsonl"))
    rows = [json.loads(line) for line in original_stream.read_text().splitlines()]
    closed = next(row for row in rows if row["event"] == "closed_environment")
    if damage == "missing":
        closed.pop("terminal_environment")
    else:
        closed["terminal_environment"]["COLUMNS"] = str(
            int(closed["terminal_environment"]["COLUMNS"]) + 1
        )
    stream = view / "process-counters/before" / original_stream.parent.name / original_stream.name
    stream.parent.mkdir(parents=True)
    stream.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(RuntimeError, match="terminal metadata differ"):
        snapshot_directories(view, "before")
