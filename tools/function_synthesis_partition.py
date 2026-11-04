# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete native Shannon partition proofs

"""Prove an original all-output miter using a complete independently checked input partition."""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeAlias

from tools.function_synthesis_aiger import AigerInterface, check_aiger_header, digest
from tools.function_synthesis_netlist import object_mapping
from tools.function_synthesis_toolchain import SynthesisToolchain, SynthesisToolError

Cofactor: TypeAlias = tuple[tuple[int, int], ...]
Leaf: TypeAlias = tuple[str, Cofactor]
_DIRECTORY = "input-partition"
_RECORD = "partition-proof.json"
_MAX_NODES = 256
_COHORT_SECONDS = 3300
_ORDER_REASON = (
    "Split original native input positions from last to first to simplify "
    "saturated branches before fractional detail."
)
_COLLAPSE = (
    "read_aiger cofactor.aig; collapse -B 1000000 -r -v; "
    "strash; write_aiger collapsed.aig; print_stats"
)


def _width(input_bits: int) -> None:
    """Require a supported complete independent primary input geometry."""
    if type(input_bits) is not int or not 1 <= input_bits <= 128:
        raise ValueError("An input partition requires between one and 128 original inputs")


def _restriction(prefix: str, assignments: Cofactor, input_bits: int) -> None:
    """Compare a declared prefix against actual native variable identities and constants."""
    if (
        len(prefix) > input_bits
        or re.fullmatch(r"[01]*", prefix) is None
        or assignments != tuple((input_bits - 1 - i, int(bit)) for i, bit in enumerate(prefix))
    ):
        raise ValueError("A native cofactor differs from its declared original-input prefix")


def verify_partition(leaves: Sequence[Leaf], input_bits: int) -> None:
    """Verify native leaf restrictions cover every original assignment exactly once.

    Parameters
    ----------
    leaves : sequence of (str, tuple of (int, int))
        Prefixes paired with original input indices and constants parsed from native commands.
    input_bits : int
        Complete original independent primary input count.

    Notes
    -----
    Numeric interval sorting is independent of builder traversal and decisions.

    Raises
    ------
    ValueError
        For a missing, duplicate, overlapping, malformed or swapped restriction.

    """
    _width(input_bits)
    intervals: list[tuple[int, int]] = []
    for prefix, assignments in leaves:
        _restriction(prefix, assignments, input_bits)
        size = 1 << (input_bits - len(prefix))
        start = int(prefix or "0", 2) * size
        intervals.append((start, start + size))
    next_start = 0
    for start, end in sorted(intervals):
        if start != next_start:
            raise ValueError("Native leaf restrictions leave a gap or overlap")
        next_start = end
    if next_start != 1 << input_bits:
        raise ValueError("Native leaf restrictions do not cover the entire original domain")


def bounded_bdd_incomplete(log: Path) -> bool:
    """Recognize only the existing native node cap or explicit deadline diagnostic.

    Parameters
    ----------
    log : pathlib.Path
        Retained actual ABC output from a bounded global or leaf collapse.

    Returns
    -------
    bool
        Whether the native collapse explicitly exhausted its existing limit.

    """
    text = log.read_text(encoding="utf-8")
    return (
        "The number of live nodes reached 1000000.\n" in text
        and "Constructing global BDDs is aborted.\n" in text
        and "Error: Collapsing has failed.\n" in text
    ) or "timeout: sending signal KILL to command" in text


def _interface(input_bits: int) -> AigerInterface:
    """Describe every original independent input and the single all-output difference."""
    return AigerInterface(
        frozenset(f"original_input_{i}" for i in range(input_bits)),
        frozenset({"all_original_output_differences_or"}),
    )


def _zero(path: Path, input_bits: int) -> bool:
    """Read the native constant while retaining every original primary input."""
    check_aiger_header(path, _interface(input_bits))
    with path.open("rb") as stream:
        fields, output = stream.readline(256).split(), stream.readline(256)
    return int(fields[-1]) == 0 and output == b"0\n"


def _bdd_zero(directory: Path, input_bits: int) -> bool:
    """Require a completed bounded BDD before admitting its retained constant-zero graph."""
    log = directory / "bdd.log"
    text = log.read_text(encoding="utf-8")
    if bounded_bdd_incomplete(log):
        return False
    markers = re.findall(
        r"^Shared BDD size =[ \t]+([0-9]+) nodes\.[ \t]+"
        r"BDD construction time =[ \t]+[0-9.]+ sec$",
        text,
        re.M,
    )
    if (
        len(markers) != 1
        or not 0 < int(markers[0]) <= 1000000
        or re.search(r"^(?:ERROR|Error|Aborting|Aborted|\*\* cmd error)\b", text, re.M)
    ):
        raise ValueError("A partition leaf has no completed bounded native BDD")
    if not _zero(directory / "collapsed.aig", input_bits):
        raise ValueError("A native partition leaf has a nonzero original-output difference")
    return True


def _native_inputs(log: Path, input_bits: int) -> tuple[str, ...]:
    """Bind cofactor operands to every actual original native primary-input name and index."""
    lines = re.findall(r"^Primary inputs \(([0-9]+)\):[ \t]*(.*)$", log.read_text(), re.M)
    if len(lines) != 1 or int(lines[0][0]) != input_bits:
        raise ValueError("The native cofactor omitted its complete original input order")
    pairs = [token.split("=", 1) for token in lines[0][1].split()]
    if (
        len(pairs) != input_bits
        or any(len(pair) != 2 or not pair[0].isdigit() for pair in pairs)
        or [int(pair[0]) for pair in pairs] != list(range(input_bits))
        or len({pair[1] for pair in pairs}) != input_bits
        or any(re.fullmatch(r"[^\s;]+", pair[1]) is None for pair in pairs)
    ):
        raise ValueError("The original native input names or indices changed")
    return tuple(pair[1] for pair in pairs)


def _script(prefix: str, inputs: tuple[str, ...]) -> str:
    """Build an exact cofactor from the original miter, preserving all original input slots."""
    return (
        "read_aiger ../../singleton.aig; logic; "
        + "".join(f"cof {inputs[-i - 1]} {bit}; " for i, bit in enumerate(prefix))
        + "strash; write_aiger cofactor.aig; print_io"
    )


def _command(path: Path, tools: SynthesisToolchain, timeout: float) -> str:
    """Read the actual bounded native invocation without trusting a recorded prefix."""
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, list)
        or len(value) != 11
        or value[:-1]
        != [
            str(tools.limits),
            "--core=0",
            "--",
            str(tools.deadline),
            "--signal=KILL",
            "--verbose",
            str(timeout),
            str(tools.abc),
            "-s",
            "-c",
        ]
        or not isinstance(value[-1], str)
    ):
        raise ValueError("A partition native command or its original deadline changed")
    return value[-1]


def _assignments(script: str, inputs: tuple[str, ...]) -> Cofactor:
    """Parse actual cofactor arguments independently from the builder's declared prefix."""
    tokens = script.split("; ")
    if tokens[:2] != ["read_aiger ../../singleton.aig", "logic"] or tokens[-3:] != [
        "strash",
        "write_aiger cofactor.aig",
        "print_io",
    ]:
        raise ValueError("A native cofactor does not start at the original complete miter")
    values: list[tuple[int, int]] = []
    for token in tokens[2:-3]:
        match = re.fullmatch(r"cof ([^\s;]+) ([01])", token)
        if match is None:
            raise ValueError("A native cofactor contains an unrecognized transformation")
        values.append((inputs.index(match[1]), int(match[2])))
    return tuple(values)


def _sources() -> dict[str, str]:
    """Bind the partition protocol, geometry reader and native execution implementation."""
    return {
        name: digest(Path(__file__).with_name(name))
        for name in (
            "function_synthesis_partition.py",
            "function_synthesis_aiger.py",
            "function_synthesis_netlist.py",
            "function_synthesis_toolchain.py",
        )
    }


def _completed(
    directory: Path,
    tools: SynthesisToolchain,
    input_bits: int,
    timestamp: str,
    timeout: float,
    rows: list[object],
    elapsed: float,
    cohort_seconds: float,
) -> dict[str, object]:
    """Reconstruct every native restriction and independently verify the complete frontier."""
    _width(input_bits)
    parsed = datetime.fromisoformat(timestamp)
    if (
        parsed.tzinfo != UTC
        or parsed.isoformat() != timestamp
        or isinstance(timeout, bool)
        or not math.isfinite(timeout)
        or not 0 < timeout <= 90
        or not math.isfinite(elapsed)
        or not 0 < elapsed <= cohort_seconds <= _COHORT_SECONDS
        or not rows
        or len(rows) > _MAX_NODES
    ):
        raise ValueError("A partition record has an invalid time, budget or node inventory")
    check_aiger_header(directory / "singleton.aig", _interface(input_bits))
    pending = [""]
    leaves: list[Leaf] = []
    reconstructed: list[dict[str, object]] = []
    partition = directory / _DIRECTORY
    inputs: tuple[str, ...] = ()
    for index, value in enumerate(rows):
        row = object_mapping(value)
        prefix = row.get("prefix")
        if not pending or not isinstance(prefix, str) or prefix != pending.pop():
            raise ValueError("A native partition omits or reorders an original split branch")
        node = partition / f"node-{index:03d}"
        if node.is_symlink() or not node.is_dir():
            raise ValueError("A partition node must be an original regular directory")
        actual_inputs = _native_inputs(node / "cofactor.log", input_bits)
        if inputs and actual_inputs != inputs:
            raise ValueError("The native cofactor changed the original input order")
        inputs = actual_inputs
        assignments = _assignments(_command(node / "cofactor.command.json", tools, timeout), inputs)
        _restriction(prefix, assignments, input_bits)
        cofactor_log = (node / "cofactor.log").read_text(encoding="utf-8")
        if re.search(r"^(?:ERROR|Error|\*\* cmd error)\b", cofactor_log, re.M):
            raise ValueError("The original native cofactor command failed")
        zero = _zero(node / "cofactor.aig", input_bits)
        status, code = "not-attempted", 0
        if not zero and len(prefix) >= max(0, input_bits - 12):
            if _command(node / "bdd.command.json", tools, timeout) != _COLLAPSE:
                raise ValueError("A leaf BDD did not compare its original complete cofactor")
            zero = _bdd_zero(node, input_bits)
            status = "zero" if zero else "incomplete"
            native_code = row.get("bdd_returncode", 0)
            if type(native_code) is not int or (
                native_code != 0
                and not (
                    native_code in {-9, 124, 137}
                    and "timeout: sending signal KILL to command" in (node / "bdd.log").read_text()
                )
            ):
                raise ValueError("A leaf native status differs from its retained outcome")
            code = native_code
        if zero:
            leaves.append((prefix, assignments))
        elif len(prefix) < input_bits:
            pending.extend([prefix + "1", prefix + "0"])
        else:
            raise ValueError("An original input assignment has no complete zero proof")
        actual = {"prefix": prefix, "bdd_status": status, "bdd_returncode": code}
        if row != actual:
            raise ValueError("The native partition record differs from its actual proof artifacts")
        reconstructed.append(actual)
    if pending:
        raise ValueError("The native input partition remains incomplete")
    verify_partition(leaves, input_bits)
    artifacts: dict[str, str] = {}
    for path in partition.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("Partition proof artifacts cannot be linked or special")
        if path.is_file() and path != partition / _RECORD:
            artifacts[path.relative_to(directory).as_posix()] = digest(path)
    return {
        "schema_version": "sc-neurocore.complete-input-partition.v1",
        "recorded_at": timestamp,
        "input_bits": input_bits,
        "latches": 0,
        "all_output_differences_or": True,
        "native_seconds": timeout,
        "cohort_seconds": cohort_seconds,
        "elapsed_seconds": elapsed,
        "maximum_nodes": _MAX_NODES,
        "live_bdd_node_limit": 1000000,
        "free_inputs_before_bdd": 12,
        "variable_order": list(range(input_bits - 1, -1, -1)),
        "order_reason": _ORDER_REASON,
        "covered_assignments": 1 << input_bits,
        "leaf_count": len(leaves),
        "leaf_prefixes": [p for p, _ in leaves],
        "nodes": reconstructed,
        "singleton_sha256": digest(directory / "singleton.aig"),
        "producer_sha256": _sources(),
        "toolchain_sha256": tools.identity,
        "artifact_sha256": dict(sorted(artifacts.items())),
    }


def read_partitioned_bdd(
    directory: Path,
    tools: SynthesisToolchain,
    input_bits: int,
) -> dict[str, object]:
    """Read complete native partition evidence without executing or restamping a proof.

    Parameters
    ----------
    directory : pathlib.Path
        Original BDD export directory containing singleton.aig and input-partition.
    tools : SynthesisToolchain
        Currently inspected native executables and technology-library contents.
    input_bits : int
        Complete original primary input geometry.

    Returns
    -------
    dict of str to object
        Independently reconstructed source-bound complete partition record.

    Raises
    ------
    OSError, ValueError
        For incomplete, stale, altered, linked or unproved native evidence.

    """
    path = directory / _DIRECTORY / _RECORD
    if directory.is_symlink() or path.parent.is_symlink() or path.is_symlink():
        raise ValueError("A partition proof must retain its original regular directory")
    tools.verify()
    text = path.read_text(encoding="utf-8")
    value = object_mapping(json.loads(text))
    timestamp, timeout, rows = (
        value.get("recorded_at"),
        value.get("native_seconds"),
        value.get("nodes"),
    )
    if not isinstance(timestamp, str) or type(timeout) is not float or not isinstance(rows, list):
        raise ValueError("A partition proof lacks its original acquisition parameters")
    elapsed, cohort = value.get("elapsed_seconds"), value.get("cohort_seconds")
    if type(elapsed) is not float or type(cohort) is not float:
        raise ValueError("A partition proof lacks its actual whole-cohort elapsed time")
    expected = _completed(directory, tools, input_bits, timestamp, timeout, rows, elapsed, cohort)
    if text != json.dumps(expected, sort_keys=True, indent=2) + "\n":
        raise ValueError("Partition proof bindings differ from the original acquisition")
    return expected


def prove_partitioned_bdd(
    directory: Path,
    tools: SynthesisToolchain,
    input_bits: int,
    *,
    timeout: float = 90.0,
    cohort_seconds: float = 3300.0,
) -> None:
    """Prove a full original-input partition under unchanged native and total limits.

    Parameters
    ----------
    directory : pathlib.Path
        Existing original complete BDD directory containing the validated singleton.
    tools : SynthesisToolchain
        Inspected native tools; the caller supplies CPU and memory containment.
    input_bits : int
        All independent original primary inputs, including every original operand.
    timeout : float, optional
        Per-command ceiling, positive and at most ninety seconds.
    cohort_seconds : float, optional
        Remaining whole-cohort allowance, positive and at most 3,300 seconds.

    Raises
    ------
    OSError, ValueError, SynthesisToolError
        For incomplete domain coverage, a nonzero difference, changed inputs,
        unrecognized native failure or exhaustion of the original finite caps.

    """
    _width(input_bits)
    if isinstance(timeout, bool) or not math.isfinite(timeout) or not 0 < timeout <= 90:
        raise ValueError("Partition commands require a finite deadline at most ninety seconds")
    if (
        isinstance(cohort_seconds, bool)
        or not math.isfinite(cohort_seconds)
        or not 0 < cohort_seconds <= _COHORT_SECONDS
    ):
        raise ValueError("A partition requires a finite cohort allowance at most 3,300 seconds")
    timeout = float(timeout)
    cohort_seconds = float(cohort_seconds)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("A partition requires its original complete BDD directory")
    source = directory / "singleton.aig"
    check_aiger_header(source, _interface(input_bits))
    before, producer = digest(source), _sources()
    partition = directory / _DIRECTORY
    partition.mkdir(exist_ok=False)
    pending = [""]
    rows: list[dict[str, object]] = []
    inputs: tuple[str, ...] = ()
    began = time.monotonic()

    def run(script: str, node: Path, label: str) -> None:
        """Reserve the unchanged whole-cohort allowance before each actual native command."""
        if time.monotonic() - began > cohort_seconds - timeout:
            raise ValueError("The bounded native partition is incomplete")
        tools.run([str(tools.abc), "-c", script], node, node / (label + ".log"), timeout=timeout)

    while pending:
        if len(rows) >= _MAX_NODES or time.monotonic() - began > cohort_seconds - timeout:
            raise ValueError("The bounded native partition is incomplete")
        prefix = pending.pop()
        node = partition / f"node-{len(rows):03d}"
        node.mkdir()
        run(_script(prefix, inputs), node, "cofactor")
        actual_inputs = _native_inputs(node / "cofactor.log", input_bits)
        if inputs and actual_inputs != inputs:
            raise ValueError("The native cofactor changed the original input order")
        inputs = actual_inputs
        zero = _zero(node / "cofactor.aig", input_bits)
        status, code = "not-attempted", 0
        if not zero and len(prefix) >= max(0, input_bits - 12):
            try:
                run(_COLLAPSE, node, "bdd")
            except SynthesisToolError as error:
                if not error.timed_out:
                    raise
                code = error.returncode
            zero = _bdd_zero(node, input_bits)
            status = "zero" if zero else "incomplete"
        rows.append({"prefix": prefix, "bdd_status": status, "bdd_returncode": code})
        if not zero:
            if len(prefix) == input_bits:
                raise ValueError("An original input assignment has no complete zero proof")
            pending.extend([prefix + "1", prefix + "0"])
    record = _completed(
        directory,
        tools,
        input_bits,
        datetime.now(UTC).isoformat(),
        timeout,
        list(rows),
        time.monotonic() - began,
        cohort_seconds,
    )
    record["elapsed_seconds"] = time.monotonic() - began
    if (
        digest(source) != before
        or _sources() != producer
        or time.monotonic() - began > cohort_seconds
    ):
        raise ValueError("The original partition input, implementation or finite budget changed")
    with (partition / _RECORD).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, indent=2) + "\n")
    read_partitioned_bdd(directory, tools, input_bits)
