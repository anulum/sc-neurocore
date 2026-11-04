# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete original-function native BDD equivalence

"""Prove every original result over the entire independent primary input domain."""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from tools.function_synthesis_aiger import (
    AigerInterface,
    check_aiger_header,
    check_equivalent_verdict,
    digest,
    read_aiger_port_order,
)
from tools.function_synthesis_checkpoints import export_checkpoint_aigers, read_checkpoint_aigers
from tools.function_synthesis_json import native_netlist_artifacts
from tools.function_synthesis_netlist import object_mapping
from tools.function_synthesis_partition import (
    bounded_bdd_incomplete,
    prove_partitioned_bdd,
    read_partitioned_bdd,
)
from tools.function_synthesis_records import SynthesisCase
from tools.function_synthesis_toolchain import SynthesisToolchain, SynthesisToolError

_DIRECTORY = "complete-bdd-proof"
_RECORD = "proof-record.json"
_ROOT = (
    "unit.v",
    "flow.ys",
    "mapped.json",
    "mapped.v",
    "stat.json",
    "yosys.log",
    "yosys.command.json",
)
_CONSTRUCTION = (
    "&read reference-plain.aig; &miter candidate-plain.aig; "
    "&write all-differences.aig; &put; orpos; &get; &write singleton.aig"
)
_COLLAPSE = (
    "read_aiger singleton.aig; collapse -B 1000000 -r -v; "
    "strash; write_aiger collapsed.aig; print_stats"
)


def _encoded(value: object) -> str:
    """Keep the complete original record in one canonical representation."""
    return json.dumps(value, sort_keys=True, indent=2) + "\n"


def _interface(case: SynthesisCase) -> AigerInterface:
    """Preserve all independently driven original arguments and result bits."""
    return AigerInterface(
        frozenset(
            f"argument{index}[{bit}]"
            for index in range(case.input_bits // case.data_width)
            for bit in range(case.data_width)
        ),
        frozenset(f"result[{bit}]" for bit in range(case.data_width)),
    )


def _sources() -> dict[str, str]:
    """Bind the actual whole-domain proof and all admission/export dependencies."""
    return {
        name: digest(Path(__file__).with_name(name))
        for name in (
            "function_synthesis_bdd.py",
            "function_synthesis_partition.py",
            "function_synthesis_checkpoints.py",
            "function_synthesis_aiger.py",
            "function_synthesis_netlist.py",
            "function_synthesis_records.py",
            "function_synthesis_toolchain.py",
            "function_synthesis_json.py",
            "function_synthesis_exports.py",
        )
    }


def _scripts(seconds: int) -> dict[str, str]:
    """Retain exact complete-domain native normalization and BDD commands."""
    result = {"cec-help": "cec -h"}
    for role, filename in (("reference", "reference.aig"), ("candidate", "mapped.aig")):
        result[role + "-io"] = f"read_aiger {filename}; print_io"
        result[role + "-normalize"] = f"read_aiger {filename}; write_aiger {role}-plain.aig"
        result[role + "-roundtrip"] = f"cec -T {seconds} -n -v {filename} {role}-plain.aig"
    io = {key: value for key, value in result.items() if key.endswith("-io")}
    rest = {key: value for key, value in result.items() if key != "cec-help" and key not in io}
    return {
        "cec-help": "cec -h",
        **io,
        **rest,
        "all-output-miter": _CONSTRUCTION,
        "complete-bdd": _COLLAPSE,
    }


def _command(path: Path, script: str, tools: SynthesisToolchain) -> None:
    """Require the original inspected native command and its admitted outer timeout."""
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("A complete BDD command requires its actual argument vector")
    argv = [item for item in value if isinstance(item, str)]
    if (
        len(argv) != 11
        or argv[:6]
        != [
            str(tools.limits),
            "--core=0",
            "--",
            str(tools.deadline),
            "--signal=KILL",
            "--verbose",
        ]
        or argv[7:] != [str(tools.abc), "-s", "-c", script]
    ):
        raise ValueError("The retained complete BDD command changed")
    seconds = float(argv[6])
    if not math.isfinite(seconds) or not 0 < seconds <= 90:
        raise ValueError("A complete BDD command exceeds its native deadline")


def _artifacts(root: Path) -> dict[str, str]:
    """Include every actual whole-function native artifact and original mapping byte."""
    directory = root / _DIRECTORY
    names = [*_ROOT, *native_netlist_artifacts(root)]
    for path in directory.rglob("*"):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError("Complete BDD evidence contains a linked or special artifact")
        if path.is_file() and path != directory / _RECORD:
            names.append(path.relative_to(root).as_posix())
    if any((root / name).is_symlink() or not (root / name).is_file() for name in names):
        raise ValueError("Complete BDD evidence requires every original regular artifact")
    return {name: digest(root / name) for name in sorted(names)}


def _global_nodes(directory: Path, scalar: AigerInterface) -> int:
    """Require an actual completed monolithic native BDD and its constant-zero result."""
    check_aiger_header(directory / "collapsed.aig", scalar)
    log = (directory / "complete-bdd.log").read_text(encoding="utf-8")
    markers = re.findall(
        r"^Shared BDD size =[ \t]+([0-9]+) nodes\.[ \t]+"
        r"BDD construction time =[ \t]+[0-9.]+ sec$",
        log,
        re.M,
    )
    if (
        len(markers) != 1
        or not 0 < int(markers[0]) <= 1000000
        or re.search(r"^(?:ERROR|Error|Aborting|Aborted)\b", log, re.M)
    ):
        raise ValueError("Native global BDD computation did not complete")
    with (directory / "collapsed.aig").open("rb") as stream:
        header, output = stream.readline(256).split(), stream.readline(256)
    if int(header[-1]) != 0 or output != b"0\n":
        raise ValueError("The complete native BDD did not prove the all-output difference zero")
    return int(markers[0])


def _completed(
    case: SynthesisCase,
    root: Path,
    tools: SynthesisToolchain,
    timestamp: str,
    seconds: int,
    strategy: Literal["global", "shannon-partitioned"] = "global",
) -> dict[str, object]:
    """Reconstruct decisive native evidence and its full original-function binding."""
    directory = root / _DIRECTORY
    artifacts = _artifacts(root)
    read_checkpoint_aigers(case, root, tools, directory)
    parsed = datetime.fromisoformat(timestamp)
    if parsed.tzinfo != UTC or parsed.isoformat() != timestamp or not 1 <= seconds <= 60:
        raise ValueError("Complete BDD evidence has an invalid acquisition time or native budget")
    scripts = _scripts(seconds)
    for label, script in scripts.items():
        for suffix in (".log", ".command.json"):
            if f"{_DIRECTORY}/{label}{suffix}" not in artifacts:
                raise ValueError("Complete BDD evidence omitted a native command or its log")
        _command(directory / (label + ".command.json"), script, tools)
    if "[default = by name]" not in (directory / "cec-help.log").read_text(encoding="utf-8"):
        raise ValueError("The native normalization matching policy is unknown")
    interface = _interface(case)
    a = read_aiger_port_order(directory / "reference-io.log", interface)
    b = read_aiger_port_order(directory / "candidate-io.log", interface)
    if a != b:
        raise ValueError("Complete BDD comparisons require identical original named bit order")
    for role in ("reference", "candidate"):
        check_aiger_header(directory / (role + "-plain.aig"), interface)
        check_equivalent_verdict(directory / (role + "-roundtrip.log"))
    check_aiger_header(directory / "all-differences.aig", interface)
    scalar = AigerInterface(interface.inputs, frozenset({"all_original_differences_or"}))
    check_aiger_header(directory / "singleton.aig", scalar)
    nodes: int | None = None
    partition: dict[str, object] | None = None
    if strategy == "global":
        nodes = _global_nodes(directory, scalar)
    else:
        if not bounded_bdd_incomplete(directory / "complete-bdd.log"):
            raise ValueError("A Shannon completion requires its original bounded global attempt")
        proof = read_partitioned_bdd(directory, tools, case.input_bits)
        partition = {
            "recorded_at": proof["recorded_at"],
            "covered_assignments": proof["covered_assignments"],
            "leaf_count": proof["leaf_count"],
            "variable_order": proof["variable_order"],
            "order_reason": proof["order_reason"],
        }
    return {
        "schema_version": "sc-neurocore.complete-original-bdd.v2",
        "case": case.key,
        "status": "equivalent",
        "method": "complete-bdd",
        "recorded_at": timestamp,
        "normalization_seconds": seconds,
        "input_bits": case.input_bits,
        "output_bits": case.data_width,
        "latches": 0,
        "input_order": list(a[0]),
        "output_order": list(a[1]),
        "auxiliary_cut_inputs": 0,
        "all_output_differences_or": True,
        "live_bdd_node_limit": 1000000,
        "strategy": strategy,
        "remaining_bdd_nodes": nodes,
        "partition": partition,
        "construction": _CONSTRUCTION,
        "collapse": _COLLAPSE,
        "producer_sha256": _sources(),
        "toolchain_sha256": tools.identity,
        "artifact_sha256": artifacts,
    }


@dataclass(frozen=True, slots=True)
class CompleteBddMappingProof:
    """Hold actual original mapping and complete native proof artifact identities.

    Parameters
    ----------
    artifact_sha256 : tuple of (str, str)
        Every original mapping/export/proof artifact, relative to the measurement.
    method : {"complete-bdd", "complete-bdd-shannon"}, optional
        Distinguish one global BDD from a complete Shannon-partitioned native proof.

    """

    artifact_sha256: tuple[tuple[str, str], ...]
    method: Literal["complete-bdd", "complete-bdd-shannon"] = "complete-bdd"


def read_complete_bdd_mapping(
    case: SynthesisCase,
    root: Path,
    tools: SynthesisToolchain,
) -> CompleteBddMappingProof:
    """Read full native BDD proof evidence without running or restamping it.

    Parameters
    ----------
    case : SynthesisCase
        Original function and complete independent argument geometry.
    root : pathlib.Path
        Original physical acquisition with its complete BDD proof directory.
    tools : SynthesisToolchain
        Currently inspected native executables and library contents.

    Returns
    -------
    CompleteBddMappingProof
        Full current artifact identities, including the original proof record.

    Raises
    ------
    OSError, ValueError
        For missing, linked, corrupt, stale, incomplete or inconclusive evidence.

    """
    path = root / _DIRECTORY / _RECORD
    if root.is_symlink() or path.parent.is_symlink() or path.is_symlink() or not path.is_file():
        raise ValueError("Complete BDD evidence requires its original regular proof record")
    text = path.read_text(encoding="utf-8")
    record = object_mapping(json.loads(text))
    timestamp, seconds = record.get("recorded_at"), record.get("normalization_seconds")
    if not isinstance(timestamp, str) or type(seconds) is not int:
        raise ValueError("Complete BDD evidence lacks its original time or normalization budget")
    strategy = record.get("strategy")
    if strategy == "global":
        expected = _completed(case, root, tools, timestamp, seconds, "global")
    elif strategy == "shannon-partitioned":
        expected = _completed(case, root, tools, timestamp, seconds, "shannon-partitioned")
    else:
        raise ValueError("Complete BDD evidence lacks its explicit native proof method")
    if text != _encoded(expected):
        raise ValueError("Complete BDD evidence differs from its original qualified bindings")
    artifacts = _artifacts(root)
    artifacts[f"{_DIRECTORY}/{_RECORD}"] = digest(path)
    return CompleteBddMappingProof(
        tuple(sorted(artifacts.items())),
        "complete-bdd-shannon" if strategy == "shannon-partitioned" else "complete-bdd",
    )


def prove_complete_bdd_mapping(
    case: SynthesisCase,
    root: Path,
    tools: SynthesisToolchain,
    *,
    timeout: float = 90.0,
    cohort_seconds: float = 3300.0,
) -> None:
    """Prove all original results with a global BDD or a complete bounded Shannon partition.

    Parameters
    ----------
    case : SynthesisCase
        Original function and every independently driven primary input bit.
    root : pathlib.Path
        Unchanged physical mapping acquisition; failed native steps remain intact.
    tools : SynthesisToolchain
        Inspected executable and technology-library closure.
    timeout : float, optional
        Positive finite per-command limit, at most ninety seconds.
    cohort_seconds : float, optional
        Positive whole-proof allowance, at most 3,300 seconds including exports.

    Raises
    ------
    OSError, ValueError
        For changed sources, malformed geometry or a nonzero/incomplete native result.
    SynthesisToolError
        If a native command fails or reaches its unchanged deadline.

    """
    if isinstance(timeout, bool) or not math.isfinite(timeout) or not 0 < timeout <= 90:
        raise ValueError("Complete BDD proofs require an admitted finite native deadline")
    if (
        isinstance(cohort_seconds, bool)
        or not math.isfinite(cohort_seconds)
        or not 0 < cohort_seconds <= 3300
    ):
        raise ValueError(
            "Complete BDD proofs require a finite cohort allowance at most 3,300 seconds"
        )
    if cohort_seconds < 2 * timeout:
        raise ValueError("The complete native BDD cohort is incomplete")
    began = time.monotonic()
    directory = root / _DIRECTORY
    sources = _sources()
    export_checkpoint_aigers(
        case,
        root,
        tools,
        directory,
        timeout=timeout,
        cohort_seconds=cohort_seconds - (time.monotonic() - began),
    )
    before = _artifacts(root)
    seconds = max(1, min(60, int(timeout)))
    strategy: Literal["global", "shannon-partitioned"] = "global"
    for label, script in _scripts(seconds).items():
        if time.monotonic() - began > cohort_seconds - timeout:
            raise ValueError("The complete native BDD cohort is incomplete")
        try:
            tools.run(
                [str(tools.abc), "-c", script],
                directory,
                directory / (label + ".log"),
                timeout=timeout,
            )
        except SynthesisToolError as error:
            if label != "complete-bdd" or not error.timed_out:
                raise
        if label == "complete-bdd" and bounded_bdd_incomplete(directory / "complete-bdd.log"):
            prove_partitioned_bdd(
                directory,
                tools,
                case.input_bits,
                timeout=timeout,
                cohort_seconds=cohort_seconds - (time.monotonic() - began),
            )
            strategy = "shannon-partitioned"
        if label == "cec-help" and "[default = by name]" not in (
            directory / "cec-help.log"
        ).read_text(encoding="utf-8"):
            raise ValueError("The native normalization matching policy is unknown")
        if label == "candidate-io":
            interface = _interface(case)
            orders = [
                read_aiger_port_order(directory / (role + "-io.log"), interface)
                for role in ("reference", "candidate")
            ]
            if orders[0] != orders[1]:
                raise ValueError("The complete original named input/output order differs")
        if label.endswith("-roundtrip"):
            check_equivalent_verdict(directory / (label + ".log"))
    record = _completed(case, root, tools, datetime.now(UTC).isoformat(), seconds, strategy)
    after = _artifacts(root)
    if (
        any(after.get(name) != value for name, value in before.items())
        or _sources() != sources
        or time.monotonic() - began > cohort_seconds
    ):
        raise ValueError("The original full native exports or proof implementation changed")
    with (directory / _RECORD).open("x", encoding="utf-8") as stream:
        stream.write(_encoded(record))
    read_complete_bdd_mapping(case, root, tools)
