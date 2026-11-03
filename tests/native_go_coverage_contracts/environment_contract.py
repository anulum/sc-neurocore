# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual inherited ABI child environment custody contracts

"""Read real parent/child declarations and refuse damaged copies of their protocol."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import cast

import pytest

from tools.go_coverage_environment import (
    child_declaration,
    environment_digest,
    environment_mapping,
    read_child_environment,
    validate_child_environment,
)


@dataclass(frozen=True)
class ActualDeclaration:
    """Retain one genuine original launch from the completed current native job."""

    base: dict[str, str]
    environment: dict[str, str]
    parent_pid: int
    argv: list[str]
    rows: list[dict[str, object]]


@pytest.fixture
def actual_declaration(native_job: Path) -> ActualDeclaration:
    """Load the actual parent PID, audit stream and inherited environment without inventing data."""
    parent = int(json.loads((native_job / "execution.json").read_text())["pid"])
    path = native_job / "process-counters/tests" / str(parent) / "observations.jsonl"
    rows = cast(
        list[dict[str, object]], [json.loads(line) for line in path.read_text().splitlines()]
    )
    row = next(item for item in rows if item["event"] == "abi_child_launch")
    return ActualDeclaration(
        cast(dict[str, str], json.loads((native_job / "environments/tests.json").read_text())),
        cast(dict[str, str], row["child_environment"]),
        parent,
        cast(list[str], row["expected_child_argv"]),
        rows,
    )


def protocol_view(tmp_path: Path, native_job: Path, actual: ActualDeclaration) -> tuple[Path, Path]:
    """Copy real protocol records into an owned fault view, preserving original raw data."""
    job = tmp_path / "environment-protocol"
    directory = job / "process-counters/tests" / str(actual.parent_pid)
    directory.mkdir(parents=True)
    shutil.copyfile(native_job / "collection.json", job / "collection.json")
    path = directory / "observations.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in actual.rows))
    return job, path


def test_real_child_environment_matches_parent_and_actual_closed_counter_stream(
    native_job: Path, actual_declaration: ActualDeclaration
) -> None:
    """The actual parent declaration matches the complete mapping captured by its real child."""
    actual = actual_declaration
    expected = read_child_environment(native_job, actual.base, actual.parent_pid, actual.argv)
    assert expected == actual.environment
    assert set(expected) == set(actual.base) | {"PYTEST_VERSION", "PYTEST_CURRENT_TEST"}
    declaration = child_declaration(native_job, actual.base, expected)
    assert declaration["child_environment"] == expected
    assert declaration["child_environment_sha256"] == environment_digest(expected)
    matched = []
    for path in (native_job / "process-counters/tests").glob("*/observations.jsonl"):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if rows[0]["ppid"] == actual.parent_pid and rows[0]["argv"] == actual.argv:
            closed = next(row for row in rows if row["event"] == "closed_environment")
            matched.append(closed)
    assert len(matched) == 1
    assert matched[0]["keys"] == sorted(expected)
    assert matched[0]["environment_sha256"] == declaration["child_environment_sha256"]
    assert (
        json.loads((native_job / "artifacts/abi-process-membership.json").read_text())[
            "actual_original_children"
        ]
        == 72
    )


@pytest.mark.parametrize("value", [[], {"HOME": 1}, {1: "value"}])
def test_non_string_or_non_mapping_environment_refuses(value: object) -> None:
    """Malformed mappings cannot be interpreted as a closed native startup declaration."""
    with pytest.raises(ValueError, match="must map strings"):
        environment_mapping(value)


@pytest.mark.parametrize(
    "damage", ["missing", "extra", "base", "version", "stage", "foreign-node", "other-case"]
)
def test_damaged_real_inherited_environment_refuses(
    native_job: Path, actual_declaration: ActualDeclaration, damage: str
) -> None:
    """Changed base values, unreviewed keys and non-original test markers fail closed."""
    actual = actual_declaration
    value = dict(actual.environment)
    if damage == "missing":
        value.pop("PYTEST_CURRENT_TEST")
    elif damage == "extra":
        value["LD_LIBRARY_PATH"] = "/unreviewed"
    elif damage == "base":
        value["HOME"] += "/different"
    elif damage == "version":
        value["PYTEST_VERSION"] += "-different"
    elif damage == "stage":
        value["PYTEST_CURRENT_TEST"] = value["PYTEST_CURRENT_TEST"].removesuffix(" (call)") + (
            " (setup)"
        )
    elif damage == "foreign-node":
        value["PYTEST_CURRENT_TEST"] += "foreign (call)"
    else:
        collection = json.loads((native_job / "collection.json").read_text())
        value["PYTEST_CURRENT_TEST"] = (
            next(
                node
                for node in collection
                if "::test_null_pointer_refusal_is_crash_fenced[" not in node
            )
            + " (call)"
        )
    with pytest.raises(ValueError):
        validate_child_environment(native_job, actual.base, value)


@pytest.mark.parametrize("damage", ["shape", "missing", "duplicate", "type"])
def test_damaged_actual_collection_refuses_child_declaration(
    native_job: Path,
    actual_declaration: ActualDeclaration,
    tmp_path: Path,
    damage: str,
) -> None:
    """A copied partial or malformed collection cannot legitimise an inherited test marker."""
    actual = actual_declaration
    job, _ = protocol_view(tmp_path, native_job, actual)
    collection = json.loads((job / "collection.json").read_text())
    if damage == "shape":
        collection = {}
    elif damage == "missing":
        collection.pop()
    elif damage == "duplicate":
        collection[-1] = collection[0]
    else:
        collection[-1] = 1
    (job / "collection.json").write_text(json.dumps(collection))
    with pytest.raises(ValueError, match="complete real 630"):
        validate_child_environment(job, actual.base, actual.environment)


@pytest.mark.parametrize(
    "damage",
    [
        "empty",
        "row",
        "registration",
        "pid",
        "missing",
        "duplicate",
        "argv",
        "parent",
        "digest",
        "environment",
    ],
)
def test_damaged_real_parent_launch_declaration_refuses(
    native_job: Path,
    actual_declaration: ActualDeclaration,
    tmp_path: Path,
    damage: str,
) -> None:
    """Only one genuine matching parent launch with its full original environment is accepted."""
    actual = actual_declaration
    job, path = protocol_view(tmp_path, native_job, actual)
    raw = json.loads(json.dumps(actual.rows))
    launch = next(row for row in raw if row["event"] == "abi_child_launch")
    if damage == "empty":
        raw = []
    elif damage == "row":
        raw[0] = "malformed record"
    elif damage == "registration":
        raw[0]["event"] = "finished"
    elif damage == "pid":
        raw[0]["pid"] += 1
    elif damage == "missing":
        raw = [row for row in raw if row["event"] != "abi_child_launch"]
    elif damage == "duplicate":
        raw.append(launch)
    elif damage == "argv":
        launch["expected_child_argv"][1] += "/different"
    elif damage == "parent":
        launch["parent_pid"] += 1
    elif damage == "digest":
        launch["child_environment_sha256"] = "0" * 64
    else:
        launch.pop("child_environment")
    path.write_text("".join(json.dumps(row) + "\n" for row in raw))
    with pytest.raises(ValueError):
        read_child_environment(job, actual.base, actual.parent_pid, actual.argv)
