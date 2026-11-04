# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native Shannon partition and independent coverage verification

"""Exercise complete native input restrictions, missing branches and retained proof integrity."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from tools.function_synthesis_partition import (
    Leaf,
    bounded_bdd_incomplete,
    prove_partitioned_bdd,
    read_partitioned_bdd,
    verify_partition,
)
from tools.function_synthesis_records import SynthesisCase
from tools.function_synthesis_toolchain import SynthesisToolchain
from tools.function_synthesis_yosys import run_function_synthesis

NativePartition = tuple[Path, SynthesisToolchain, int]


@pytest.mark.parametrize(
    ("leaves", "width"),
    [
        ([("", ())], 1),
        ([("", ())], 128),
        ([("0", ((3, 0),)), ("10", ((3, 1), (2, 0))), ("11", ((3, 1), (2, 1)))], 4),
        ([("1", ((1, 1),)), ("00", ((1, 0), (0, 0))), ("01", ((1, 0), (0, 1)))], 2),
    ],
)
def test_partition_domain_verifier_accepts_complete_native_restrictions(
    leaves: list[Leaf],
    width: int,
) -> None:
    """Verify numeric coverage independently of depth-first order or equal-depth leaves."""
    verify_partition(leaves, width)


@pytest.mark.parametrize(
    ("leaves", "width"),
    [
        ([], 4),
        ([("0", ((3, 0),))], 4),
        ([("1", ((3, 1),))], 4),
        ([("0", ((3, 0),)), ("0", ((3, 0),)), ("1", ((3, 1),))], 4),
        ([("", ()), ("0", ((3, 0),))], 4),
        ([("00", ((3, 0), (2, 0))), ("1", ((3, 1),))], 4),
        ([("0", ((3, 1),)), ("1", ((3, 0),))], 4),
        ([("0", ((2, 0),)), ("1", ((2, 1),))], 4),
        ([("0", ((3, 0), (2, 0))), ("1", ((3, 1),))], 4),
        ([("x", ((3, 0),))], 4),
        ([("00000", ((3, 0), (2, 0), (1, 0), (0, 0), (-1, 0)))], 4),
    ],
)
def test_partition_domain_verifier_refuses_missing_duplicate_and_swapped_cofactors(
    leaves: list[Leaf],
    width: int,
) -> None:
    """Refuse incomplete or overlapping coverage even when a caller declares success."""
    with pytest.raises(ValueError):
        verify_partition(leaves, width)


@pytest.mark.parametrize("width", [False, 0, -1, 129])
def test_partition_domain_verifier_refuses_nonoriginal_geometry(width: int) -> None:
    """Require supported independent input counts before interpreting any restriction."""
    with pytest.raises(ValueError):
        verify_partition([("", ())], width)


@pytest.fixture(scope="module")
def native_partition(tmp_path_factory: pytest.TempPathFactory) -> NativePartition:
    """Run a real compiler function, native miter construction and complete partition API."""
    base = tmp_path_factory.mktemp("native-input-partition")
    tools = SynthesisToolchain.inspect(base / "inspection")
    case = SynthesisCase("ceil", 16, 8)
    mapped = base / "mapping"
    run_function_synthesis(case, mapped, tools)
    directory = base / "proof"
    directory.mkdir()
    for name in ("reference.aig", "mapped.aig"):
        shutil.copyfile(mapped / name, directory / name)
    tools.run(
        [
            str(tools.abc),
            "-c",
            "&read reference.aig; &miter mapped.aig; &put; orpos; &get; &write singleton.aig",
        ],
        directory,
        directory / "miter.log",
        timeout=90.0,
    )
    prove_partitioned_bdd(directory, tools, case.input_bits)
    return directory, tools, case.input_bits


def test_native_partition_readback_preserves_original_bytes_and_times(
    native_partition: NativePartition,
) -> None:
    """Read a source-bound real native proof without replaying or restamping its commands."""
    directory, tools, width = native_partition
    before = {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in directory.rglob("*")
        if path.is_file()
    }
    record = read_partitioned_bdd(directory, tools, width)
    assert before == {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in before
    }
    assert record["covered_assignments"] == 1 << width
    assert record["variable_order"] == list(range(width - 1, -1, -1))
    assert "saturated branches" in str(record["order_reason"])
    assert record["native_seconds"] == 90.0
    assert record["cohort_seconds"] == 3300
    assert record["live_bdd_node_limit"] == 1000000
    assert 0 < float(str(record["elapsed_seconds"])) <= 3300


@pytest.mark.parametrize(
    "entry",
    [
        "singleton.aig",
        "input-partition/node-000/cofactor.aig",
        "input-partition/node-000/cofactor.log",
        "input-partition/node-000/cofactor.command.json",
        "input-partition/partition-proof.json",
    ],
)
def test_real_native_partition_changed_artifacts_are_refused(
    native_partition: NativePartition,
    tmp_path: Path,
    entry: str,
) -> None:
    """Bind the original whole miter, actual native command, graph, output and final record."""
    original, tools, width = native_partition
    changed = tmp_path / "changed"
    shutil.copytree(original, changed)
    path = changed / entry
    path.write_bytes(path.read_bytes() + b"\nchanged")
    with pytest.raises(ValueError):
        read_partitioned_bdd(changed, tools, width)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("native_seconds", 90),
        ("native_seconds", 91.0),
        ("native_seconds", float("nan")),
        ("elapsed_seconds", None),
        ("elapsed_seconds", -1.0),
        ("elapsed_seconds", 3301.0),
        ("elapsed_seconds", float("inf")),
        ("recorded_at", "2026-11-02T00:00:00"),
        ("nodes", []),
        ("nodes", "omitted"),
        ("variable_order", []),
        ("order_reason", "random ordering"),
        ("covered_assignments", 0),
        ("cohort_seconds", 9999),
        ("maximum_nodes", 9999),
        ("producer_sha256", {}),
    ],
)
def test_real_native_partition_record_relaxation_is_refused(
    native_partition: NativePartition,
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    """Reject reduced obligations, enlarged budgets, stale sources and missing native execution."""
    original, tools, width = native_partition
    changed = tmp_path / "changed"
    shutil.copytree(original, changed)
    path = changed / "input-partition/partition-proof.json"
    record = json.loads(path.read_text())
    record[field] = value
    path.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    with pytest.raises(ValueError):
        read_partitioned_bdd(changed, tools, width)


def test_real_native_partition_swapped_actual_cofactor_is_refused(
    native_partition: NativePartition,
    tmp_path: Path,
) -> None:
    """Reject a new actual restriction even if its retained graph still happens to be zero."""
    original, tools, width = native_partition
    changed = tmp_path / "changed"
    shutil.copytree(original, changed)
    path = changed / "input-partition/node-000/cofactor.command.json"
    command = json.loads(path.read_text())
    command[-1] = command[-1].replace(
        "read_aiger ../../singleton.aig; logic; ",
        "read_aiger ../../singleton.aig; logic; cof pi00 1; ",
        1,
    )
    path.write_text(json.dumps(command, indent=2) + "\n")
    with pytest.raises(ValueError, match="cofactor differs"):
        read_partitioned_bdd(changed, tools, width)


@pytest.mark.parametrize("kind", ["root", "partition", "node", "record", "graph"])
def test_real_native_partition_linked_artifacts_are_refused(
    native_partition: NativePartition,
    tmp_path: Path,
    kind: str,
) -> None:
    """Refuse links at the proof root, partition, native node, record or actual graph."""
    original, tools, width = native_partition
    changed = tmp_path / "changed"
    shutil.copytree(original, changed)
    relative = {
        "root": "",
        "partition": "input-partition",
        "node": "input-partition/node-000",
        "record": "input-partition/partition-proof.json",
        "graph": "input-partition/node-000/cofactor.aig",
    }[kind]
    path = changed / relative
    saved = path.with_name(path.name + "-saved")
    path.rename(saved)
    path.symlink_to(saved, target_is_directory=saved.is_dir())
    with pytest.raises(ValueError):
        read_partitioned_bdd(changed, tools, width)


@pytest.mark.parametrize("budget", [False, 0.0, -1.0, 91.0, float("nan"), float("inf")])
def test_partition_native_deadlines_are_not_relaxed(
    native_partition: NativePartition,
    budget: float,
) -> None:
    """Refuse invalid budgets before altering the original genuine successful acquisition."""
    directory, tools, width = native_partition
    path = directory / "input-partition/partition-proof.json"
    before = path.read_bytes()
    with pytest.raises(ValueError):
        prove_partitioned_bdd(directory, tools, width, timeout=budget)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "The number of live nodes reached 1000000.\n"
            "Constructing global BDDs is aborted.\nError: Collapsing has failed.\n",
            True,
        ),
        ("/usr/bin/timeout: sending signal KILL to command 'yosys-abc'\n", True),
        ("Error: Collapsing has failed.\n", False),
        ("Networks are UNDECIDED\n", False),
        ("Shared BDD size = 1 nodes. BDD construction time = 0.01 sec\n", False),
    ],
)
def test_only_explicit_existing_bdd_limits_admit_partition_completion(
    tmp_path: Path,
    text: str,
    expected: bool,
) -> None:
    """Distinguish the existing cap and deadline from unknown errors or an unproved verdict."""
    log = tmp_path / "native.log"
    log.write_text(text)
    assert bounded_bdd_incomplete(log) is expected


def test_partition_refuses_a_real_nonzero_original_output_miter(
    native_partition: NativePartition,
    tmp_path: Path,
) -> None:
    """Find the changed output sign bit instead of accepting the zero half of its domain."""
    _, tools, width = native_partition
    directory = tmp_path / "nonzero"
    directory.mkdir()
    (directory / "fault.v").write_text(
        f"module function_unit(input [{width - 1}:0] argument0, output difference);\n"
        f"assign difference = argument0[{width - 1}];\nendmodule\n"
    )
    tools.run(
        [
            str(tools.yosys),
            "-Q",
            "-T",
            "-p",
            "read_verilog fault.v; hierarchy -check -top function_unit; "
            "aigmap; write_aiger -symbols singleton.aig",
        ],
        directory,
        directory / "fault-export.log",
        timeout=90.0,
    )
    with pytest.raises(ValueError, match="nonzero original-output difference"):
        prove_partitioned_bdd(directory, tools, width)
    assert not (directory / "input-partition/partition-proof.json").exists()
    assert (directory / "input-partition/node-000/cofactor.log").is_file()


@pytest.mark.parametrize("identity", [False, True])
def test_native_partition_nonzero_leaf_cannot_be_relabelled_complete(
    native_partition: NativePartition,
    tmp_path: Path,
    identity: bool,
) -> None:
    """Change an actual native zero graph to one while retaining the claimed success record."""
    original, tools, width = native_partition
    changed = tmp_path / "changed"
    if identity:
        changed.mkdir()
        shutil.copyfile(original / "reference.aig", changed / "reference.aig")
        tools.run(
            [
                str(tools.abc),
                "-c",
                "&read reference.aig; &miter reference.aig; &put; orpos; &get; &write singleton.aig",
            ],
            changed,
            changed / "identity-miter.log",
            timeout=90.0,
        )
        prove_partitioned_bdd(changed, tools, width)
        assert read_partitioned_bdd(changed, tools, width)["leaf_prefixes"] == [""]
    else:
        shutil.copytree(original, changed)
    record = json.loads((changed / "input-partition/partition-proof.json").read_text())
    rows = record["nodes"]
    index = next((i for i, row in enumerate(rows) if row["bdd_status"] == "zero"), None)
    if index is None:
        index = len(rows) - 1
        name = "cofactor.aig"
    else:
        name = "collapsed.aig"
    path = changed / "input-partition" / f"node-{index:03d}" / name
    header, output, rest = path.read_bytes().split(b"\n", 2)
    assert header.split()[-1] == b"0" and output == b"0"
    path.write_bytes(header + b"\n1\n" + rest)
    with pytest.raises((OSError, ValueError)):
        read_partitioned_bdd(changed, tools, width)


@pytest.mark.parametrize(
    ("entry", "before", "after"),
    [
        ("node-000/cofactor.log", "Primary inputs (16)", "Primary inputs (15)"),
        ("node-000/cofactor.log", "0=pi00", "0=pi01"),
        ("node-000/cofactor.log", "0=pi00 1=pi01", "1=pi00 0=pi01"),
        ("node-000/cofactor.log", "0=pi00", "0=pi00;quit"),
        ("node-001/cofactor.log", "pi00", "renamed_original_input"),
        ("node-000/cofactor.log", "Latches (0):", "Latches (0):\nERROR: incomplete cofactor"),
        ("leaf/bdd.log", "Shared BDD size =", "Omitted BDD completion ="),
        ("leaf/bdd.log", "Shared BDD size =", "ERROR: collapse failed\nShared BDD size ="),
        (
            "leaf/bdd.log",
            "Shared BDD size =",
            "The number of live nodes reached 1000000.\n"
            "Constructing global BDDs is aborted.\nError: Collapsing has failed.\n"
            "Shared BDD size =",
        ),
        (
            "leaf/bdd.log",
            "Shared BDD size =",
            "timeout: sending signal KILL to command\nShared BDD size =",
        ),
    ],
)
def test_native_partition_refuses_incomplete_native_interfaces_and_verdicts(
    native_partition: NativePartition,
    tmp_path: Path,
    entry: str,
    before: str,
    after: str,
) -> None:
    """Refuse altered original input identities, incomplete cofactors and uncompleted leaf proofs."""
    original, tools, width = native_partition
    changed = tmp_path / "changed"
    shutil.copytree(original, changed)
    partition = changed / "input-partition"
    if entry.startswith("leaf/"):
        entry = str(next(partition.glob("node-*/bdd.log")).relative_to(partition))
    path = partition / entry
    text = path.read_text()
    assert before in text
    path.write_text(text.replace(before, after))
    with pytest.raises(ValueError):
        read_partitioned_bdd(changed, tools, width)


@pytest.mark.parametrize("mode", ["missing", "duplicate", "reordered", "status", "code", "boolean"])
def test_native_partition_replays_every_actual_branch_and_exit_status(
    native_partition: NativePartition,
    tmp_path: Path,
    mode: str,
) -> None:
    """Reject a dropped sibling, repeated branch, reordered node or changed native outcome."""
    original, tools, width = native_partition
    changed = tmp_path / "changed"
    shutil.copytree(original, changed)
    path = changed / "input-partition/partition-proof.json"
    record = json.loads(path.read_text())
    rows = record["nodes"]
    leaf = next(row for row in rows if row["bdd_status"] == "zero")
    if mode == "missing":
        rows.pop()
    elif mode == "duplicate":
        rows.append(rows[-1])
    elif mode == "reordered":
        rows[0], rows[1] = rows[1], rows[0]
    elif mode == "status":
        rows[0]["bdd_status"] = "zero"
    else:
        leaf["bdd_returncode"] = False if mode == "boolean" else -9
    path.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    with pytest.raises(ValueError):
        read_partitioned_bdd(changed, tools, width)


@pytest.mark.parametrize(
    "script",
    [
        None,
        "read_aiger substitute.aig; print_io",
        "read_aiger ../../singleton.aig; logic; quit; strash; write_aiger cofactor.aig; print_io",
    ],
)
def test_native_partition_refuses_changed_actual_command_vectors(
    native_partition: NativePartition,
    tmp_path: Path,
    script: str | None,
) -> None:
    """Require the inspected executable vector and permitted original-miter cofactor commands."""
    original, tools, width = native_partition
    changed = tmp_path / "changed"
    shutil.copytree(original, changed)
    path = changed / "input-partition/node-000/cofactor.command.json"
    command = json.loads(path.read_text())
    if script is None:
        command.pop(0)
    else:
        command[-1] = script
    path.write_text(json.dumps(command, indent=2) + "\n")
    with pytest.raises(ValueError):
        read_partitioned_bdd(changed, tools, width)


@pytest.mark.parametrize("linked", [False, True])
def test_native_partition_acquisition_requires_an_original_directory(
    native_partition: NativePartition,
    tmp_path: Path,
    linked: bool,
) -> None:
    """Refuse missing or linked acquisition roots before running any native command."""
    original, tools, width = native_partition
    target = tmp_path / "unavailable"
    if linked:
        target.symlink_to(original, target_is_directory=True)
    with pytest.raises(ValueError, match="original complete BDD directory"):
        prove_partitioned_bdd(target, tools, width)


def test_native_partition_rejects_an_extra_link_after_complete_leaf_verification(
    native_partition: NativePartition,
    tmp_path: Path,
) -> None:
    """Refuse an additional linked artifact even when every original native leaf remains intact."""
    original, tools, width = native_partition
    changed = tmp_path / "changed"
    shutil.copytree(original, changed)
    (changed / "input-partition/unrecorded-native-graph").symlink_to(original / "singleton.aig")
    with pytest.raises(ValueError, match="linked or special"):
        read_partitioned_bdd(changed, tools, width)
