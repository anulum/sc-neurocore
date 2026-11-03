# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native Go process and counter conservation

"""Reconcile actual observer records with complete original ABI consumption.

These checks consume observations; they do not manufacture process exit codes.
The unchanged ABI tests assert each crash-fenced child's actual zero status.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Container
from dataclasses import dataclass
import json
from pathlib import Path
from typing import cast

from tools.go_coverage_images import Image, read_images
from tools.go_coverage_environment import (
    environment_digest,
    environment_mapping,
    read_child_environment,
)

Record = dict[str, object]


def records(path: Path) -> list[Record]:
    """Read a complete process stream, refusing missing or foreign terminal facts."""
    values: list[object] = [json.loads(line) for line in path.read_text().splitlines()]
    if not values or any(not isinstance(value, dict) for value in values):
        raise ValueError("Malformed or empty native process observation")
    rows = cast(list[Record], values)
    allowed = {
        "registered",
        "closed_environment",
        "finished",
        "pre_dlopen",
        "invalid_snapshot_path",
        "snapshot",
        "retained_mojo_mapping",
        "abi_child_launch",
    }
    if any(row.get("event") not in allowed for row in rows):
        raise ValueError("Unknown native process observation event")
    if rows[0].get("event") != "registered" or rows[-1].get("event") != "finished":
        raise RuntimeError("A participating process did not finish its native observations")
    if any(
        sum(row.get("event") == name for row in rows) != 1
        for name in ("registered", "closed_environment", "finished")
    ):
        raise RuntimeError("A process registration, environment or terminal is ambiguous")
    pid = rows[0].get("pid")
    if type(pid) is not int or str(pid) != path.parent.name:
        raise RuntimeError("Process record path differs from its actual registered PID")
    if any("pid" in row and row["pid"] != pid for row in rows):
        raise RuntimeError("Process observations contain a different PID")
    return rows


def observed_roles(rows: list[Record]) -> set[tuple[object, object]]:
    """Return actual load roles without treating a manifest as consumption evidence."""
    return {
        (row.get("backend"), row.get("model")) for row in rows if row.get("event") == "pre_dlopen"
    }


def process_snapshots(process: Path, rows: list[Record], images: dict[Path, Image]) -> list[Path]:
    """Require both genuine cumulative snapshots for each actual loaded Go image."""
    loaded: dict[str, Image] = {}
    for row in rows:
        if row.get("event") != "pre_dlopen":
            continue
        image = images.get(Path(str(row.get("path"))))
        if image is None or (row.get("sha256"), row.get("backend"), row.get("model")) != (
            image.sha256,
            image.backend,
            image.model,
        ):
            raise RuntimeError("Observed native load differs from the declared actual image")
        loaded[str(image.path)] = image
    go = {name: image for name, image in loaded.items() if image.backend == "go"}
    refused = [row for row in rows if row.get("event") == "invalid_snapshot_path"]
    if Counter(str(row.get("path")) for row in refused) != Counter(go.keys()):
        raise RuntimeError("Each actual Go load must exercise one native path refusal")
    if any(row.get("status") != 1 for row in refused):
        raise RuntimeError("An invalid native snapshot path was not actually refused")
    snapshots = [row for row in rows if row.get("event") == "snapshot"]
    expected = {(name, capture) for name in go for capture in ("first", "final")}
    actual = [(str(row.get("path")), row.get("capture")) for row in snapshots]
    if len(actual) != len(expected) or set(actual) != expected:
        raise RuntimeError("Actual Go loads and cumulative native snapshots disagree")
    result = []
    for row in snapshots:
        image = go[str(row["path"])]
        destination = process / image.model / str(row["capture"])
        if (
            row.get("status") != 0
            or row.get("sha256") != image.sha256
            or row.get("destination") != str(destination)
            or destination.is_symlink()
            or not destination.resolve(strict=True).is_relative_to(process)
        ):
            raise RuntimeError("Native snapshot result, identity or destination differs")
        names = sorted(child.name for child in destination.iterdir())
        if (
            row.get("files") != names
            or not row.get("maps")
            or not list(destination.glob("covmeta.*"))
            or not list(destination.glob("covcounters.*"))
        ):
            raise RuntimeError("Actual native snapshot metadata, counters or mappings are missing")
        result.append(destination)
    mojo = {name for name, image in loaded.items() if image.backend == "mojo"}
    retained = [row for row in rows if row.get("event") == "retained_mojo_mapping"]
    if Counter(str(row.get("path")) for row in retained) != Counter(mojo) or any(
        not row.get("maps") for row in retained
    ):
        raise RuntimeError("Actual Mojo loads and retained executable mappings disagree")
    return result


def verify_abi_processes(job: Path, processes: dict[int, list[Record]]) -> None:
    """Conserve all original launches, child origins and consumed image roles."""
    execution = json.loads((job / "execution.json").read_text())
    collection = json.loads((job / "collection-process.json").read_text())
    parent_pid, collector_pid = execution["pid"], collection["pid"]
    if parent_pid == collector_pid or parent_pid not in processes or collector_pid not in processes:
        raise RuntimeError(
            "Actual ABI execution and collection processes are missing or duplicated"
        )
    parent, collector = processes[parent_pid], processes[collector_pid]
    children = [rows for pid, rows in processes.items() if pid not in {parent_pid, collector_pid}]
    launches = [row for row in parent if row.get("event") == "abi_child_launch"]
    expected = [json.dumps(row.get("expected_child_argv")) for row in launches]
    actual = [json.dumps(rows[0].get("argv")) for rows in children]
    if (
        len(children) != 72
        or len(expected) != 72
        or len(set(expected)) != 72
        or sorted(expected) != sorted(actual)
        or any(rows[0].get("ppid") != parent_pid for rows in children)
        or any(row.get("parent_pid") != parent_pid for row in launches)
    ):
        raise RuntimeError("Actual original crash-fenced launches and child registrations disagree")
    roles = {
        (backend, model) for backend in ("go", "mojo") for model in ("mat", "sc_resetting_mat")
    }
    if observed_roles(parent) != roles or observed_roles(collector):
        raise RuntimeError("Full ABI parent or collection-only load membership differs")
    if any(row.get("event") == "abi_child_launch" for row in collector):
        raise RuntimeError("Collection-only process unexpectedly launched an ABI child")
    for rows in children:
        argv = rows[0].get("argv")
        if not isinstance(argv, list) or len(argv) != 7:
            raise RuntimeError("Original ABI child argument geometry changed")
        if observed_roles(rows) != {(argv[3], argv[2])}:
            raise RuntimeError("Original ABI child did not consume its declared image")
    if (
        execution["pytest_exit"] != 0
        or execution["junit_cases"] != 630
        or execution["rejected"]
        or execution["other_reports"]
        or len(execution["passed"]) != 630
    ):
        raise RuntimeError("Original ABI child exit assertions were not all successful")
    with (job / "artifacts/abi-process-membership.json").open("x") as stream:
        stream.write(
            json.dumps(
                {
                    "parent_pid": parent_pid,
                    "collector_pid": collector_pid,
                    "actual_original_children": len(children),
                    "matched_launches": len(launches),
                    "child_exit_evidence": (
                        "unchanged null-pointer tests assert actual subprocess.returncode == 0"
                    ),
                    "observer_provides_independent_child_exit_codes": False,
                },
                indent=2,
            )
            + "\n"
        )


@dataclass(frozen=True)
class ProcessScope:
    """Bind parser inputs to a real job, validated phase and its original declarations."""

    job: Path
    phase: str
    interpreter: dict[str, str]
    environment: dict[str, str]


def validate_process_header(
    scope: ProcessScope, rows: list[Record], seen_pids: Container[int]
) -> int:
    """Validate parsed real process headers without replacing their source observations.

    Parameters
    ----------
    scope : ProcessScope
        Original job declarations and the validated before/tests phase.
    rows : list of dict
        Nonempty, parser-validated actual records, or an explicitly damaged copy.
    seen_pids : container of int
        Actual earlier process IDs already consumed in this phase.

    Returns
    -------
    int
        The unchanged registered process ID after full identity/environment checks.

    Raises
    ------
    RuntimeError, ValueError
        Process identity, child ancestry or the complete closed declaration differs.
    """
    pid = cast(int, rows[0]["pid"])
    if pid in seen_pids or rows[0].get("interpreter") != scope.interpreter:
        raise RuntimeError("Actual process PID or interpreter custody differs")
    expected = scope.environment
    argv = rows[0].get("argv")
    if scope.phase == "tests" and isinstance(argv, list) and argv[:1] == ["-c"]:
        parent_pid = rows[0].get("ppid")
        if type(parent_pid) is not int or not all(isinstance(value, str) for value in argv):
            raise RuntimeError("Original ABI child parent or argv geometry differs")
        expected = read_child_environment(scope.job, expected, parent_pid, cast(list[str], argv))
    closed = next(row for row in rows if row.get("event") == "closed_environment")
    if (
        closed.get("keys") != sorted(expected)
        or closed.get("environment_sha256") != environment_digest(expected)
        or closed.get("terminal_environment")
        != {key: expected.get(key) for key in ("COLUMNS", "LINES")}
    ):
        raise RuntimeError("Recorded consumer environment keys, digest or terminal metadata differ")
    return pid


def validate_snapshot_membership(
    job: Path, phase: str, processes: dict[int, list[Record]], snapshots: list[Path]
) -> None:
    """Conserve a complete snapshot set after the public reader validates its phase.

    Parameters
    ----------
    job : pathlib.Path
        Actual job supplying original ABI execution and collection evidence.
    phase : str
        Already validated before/tests selector from the public snapshot reader.
    processes : dict
        Real parsed process streams, or damaged copies used only for refusal checks.
    snapshots : list of pathlib.Path
        Actual native snapshot paths; no path or counter is manufactured here.

    Raises
    ------
    RuntimeError
        Snapshot uniqueness or the phase's actual process/load membership differs.
    """
    if not snapshots or len(snapshots) != len(set(snapshots)):
        raise RuntimeError("Native snapshot directories are missing or duplicated")
    if phase == "before":
        if len(processes) != 1 or observed_roles(next(iter(processes.values()))) != {
            ("go", "mat"),
            ("go", "sc_resetting_mat"),
        }:
            raise RuntimeError("Before-work native image preload membership differs")
    else:
        verify_abi_processes(job, processes)


def snapshot_directories(job: Path, phase: str) -> list[Path]:
    """Require complete real process custody and every declared native snapshot."""
    if phase not in {"before", "tests"}:
        raise ValueError("Only complete before-work or ABI-test phases may be merged")
    root = job / "process-counters" / phase
    images = read_images(job / "images.json", job)
    expected_interpreter = json.loads((job / "python.json").read_text())
    base_environment = environment_mapping(
        json.loads((job / "environments" / (phase + ".json")).read_text())
    )
    scope = ProcessScope(job, phase, expected_interpreter, base_environment)
    processes: dict[int, list[Record]] = {}
    result = []
    for path in sorted(root.glob("*/observations.jsonl")):
        rows = records(path)
        pid = validate_process_header(scope, rows, processes)
        processes[pid] = rows
        result.extend(process_snapshots(path.parent, rows, images))
    validate_snapshot_membership(job, phase, processes, result)
    return result
