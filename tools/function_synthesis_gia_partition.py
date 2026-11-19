# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete native comparison by output groups

"""Verify all native outputs through exact structural identities and bounded comparisons."""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from tools.function_synthesis_aiger import AigerInterface, check_aiger_header, digest
from tools.function_synthesis_checkpoint_timeout import validate_checkpoint_command
from tools.function_synthesis_gia_protocol import _check_log, _needs_further_proof
from tools.function_synthesis_json import MAX_NETLIST_BYTES
from tools.function_synthesis_partition import _native_inputs
from tools.function_synthesis_retention import _MAX_BYTES
from tools.function_synthesis_toolchain import SynthesisToolchain, SynthesisToolError

PARTITION_DIRECTORY = "native-output-comparisons"
_MAX_COMPARISONS = 256
_ROLES = ("reference", "candidate")


def _regular(path: Path) -> None:
    """Require a directly retained and bounded native comparison artifact."""
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_NETLIST_BYTES:
        raise ValueError("A partitioned comparison requires a bounded regular artifact")


def _delta(stream: Iterator[int]) -> int:
    """Read one bounded native binary AIGER delta without changing its polarity."""
    value = 0
    for shift in range(0, 35, 7):
        byte = next(stream, -1)
        if byte < 0:
            break
        value |= (byte & 127) << shift
        if byte < 128:
            return value
    raise ValueError("A partitioned comparison has an invalid native AIGER literal")


def _write_delta(target: bytearray, value: int) -> None:
    """Append an unsigned AIGER gate delta in its native binary encoding."""
    while value > 127:
        target.append((value & 127) | 128)
        value >>= 7
    target.append(value)


@dataclass(frozen=True, slots=True)
class _Aiger:
    """Preserve the complete native binary graph and its ordered output identities."""

    inputs: int
    literals: tuple[int, ...]
    gates: tuple[tuple[int, int], ...]
    identities: tuple[int, ...]

    @classmethod
    def parse(
        cls, path: Path, interface: AigerInterface, intern: dict[tuple[int, int], int]
    ) -> _Aiger:
        """Canonicalize the same named independent inputs and AND edges for both graphs."""
        _regular(path)
        check_aiger_header(path, interface)
        lines = path.read_bytes().split(b"\n", len(interface.outputs) + 1)
        header = tuple(lines[0].split())
        if len(header) != 6 or header[0] != b"aig" or len(lines) != len(interface.outputs) + 2:
            raise ValueError("A partitioned comparison requires a complete native binary AIGER")
        maximum, inputs, latches, outputs, gates = map(int, header[1:])
        if latches or maximum != inputs + gates or outputs != len(interface.outputs):
            raise ValueError("A partitioned comparison changes native AIGER geometry")
        body = lines[-1]
        if gates > len(body) // 2:
            raise ValueError("A partitioned comparison has an incomplete native AIGER body")
        stream = iter(body)
        identities = [2 * index for index in range(inputs + 1)]
        edges = []

        def identity(literal: int) -> int:
            """Bind one original phase to the identical structural node."""
            if not 0 <= literal < 2 * len(identities):
                raise ValueError("A partitioned comparison has an invalid original literal")
            return identities[literal >> 1] ^ (literal & 1)

        for index in range(inputs + 1, maximum + 1):
            first = 2 * index - _delta(stream)
            second = first - _delta(stream)
            edges.append((first, second))
            left, right = identity(first), identity(second)
            operands = (min(left, right), max(left, right))
            identities.append(intern.setdefault(operands, 2 * (inputs + 1 + len(intern))))
        literals = tuple(map(int, lines[1:-1]))
        return cls(inputs, literals, tuple(edges), tuple(map(identity, literals)))

    def select(self, outputs: tuple[int, ...]) -> bytes:
        """Write only gates driving selected outputs while retaining every original input."""
        selected = tuple(self.literals[index] for index in outputs)
        reachable: set[int] = set()
        pending = [literal >> 1 for literal in selected]
        first_gate = self.inputs + 1
        while pending:
            index = pending.pop()
            if index >= first_gate and index not in reachable:
                reachable.add(index)
                high, low = self.gates[index - first_gate]
                pending.extend((high >> 1, low >> 1))
        ordered = sorted(reachable)
        remap = {index: first_gate + offset for offset, index in enumerate(ordered)}

        def literal_at(literal: int) -> int:
            """Keep input indices and literal phases when renumbering reachable gates."""
            index = literal >> 1
            if index < first_gate:
                return literal
            return (remap[index] << 1) | (literal & 1)

        count = len(ordered)
        body = bytearray()
        for index in ordered:
            high, low = self.gates[index - first_gate]
            mapped_high, mapped_low = literal_at(high), literal_at(low)
            _write_delta(body, 2 * remap[index] - mapped_high)
            _write_delta(body, mapped_high - mapped_low)
        header = f"aig {self.inputs + count} {self.inputs} 0 {len(selected)} {count}\n"
        output_literals = "".join(f"{literal_at(literal)}\n" for literal in selected)
        return header.encode("ascii") + output_literals.encode("ascii") + bytes(body)


def _groups(
    reference: _Aiger, candidate: _Aiger, outputs: tuple[str, ...]
) -> tuple[tuple[int, ...], ...]:
    """Cover identical output pairs once and group the remaining pairs by original port."""
    represented: set[tuple[int, int]] = set()
    by_port: dict[str, list[int]] = {}
    for index, pair in enumerate(zip(reference.identities, candidate.identities, strict=True)):
        if pair[0] != pair[1] and pair not in represented:
            represented.add(pair)
            port = re.sub(r"\[\d+\]$", "", outputs[index])
            by_port.setdefault(port, []).append(index)
    return tuple(tuple(indices) for indices in by_port.values())


def _sign_positions(inputs: tuple[str, ...]) -> tuple[int, ...]:
    """Find the highest original bit of each independently named signed argument."""
    positions: dict[str, tuple[int, int]] = {}
    for position, name in enumerate(inputs):
        match = re.fullmatch(r"(argument[0-9]+)\[([0-9]+)\]", name)
        if match is not None:
            port, bit = match.groups()
            if int(bit) > positions.get(port, (-1, -1))[0]:
                positions[port] = int(bit), position
    if not 1 <= len(positions) <= 2:
        raise ValueError("The native output comparison has no bounded original sign partition")
    return tuple(index for _, index in positions.values())


def check_partitioned_outputs(
    reference: Path,
    candidate: Path,
    input_order: tuple[str, ...],
    output_order: tuple[str, ...],
    tools: SynthesisToolchain,
    directory: Path,
    timeout: float,
    *,
    acquire: bool,
) -> dict[str, object]:
    """Run or read complete output comparisons with exact original native subgraphs.

    Parameters
    ----------
    reference, candidate : pathlib.Path
        Original complete normalised AIGER models with an independently checked named order.
    input_order, output_order : tuple of str
        The unchanged ordered primary interface of both native models.
    tools : SynthesisToolchain
        Inspected native commands with any inherited complete-proof deadline.
    directory : pathlib.Path
        Dedicated proof directory for the complete subdivision.
    timeout : float
        Original finite native ceiling in seconds.
    acquire : bool
        Run native comparisons when true. Otherwise read the existing evidence without writes.

    Returns
    -------
    dict of str to object
        Exact original output groups, geometry and verified native artifact identities.

    Raises
    ------
    OSError, ValueError, SynthesisToolError
        If any original output or sign case is unproved, changed or outside the original bounds.

    """
    if isinstance(timeout, bool) or not math.isfinite(timeout) or not 0 < timeout <= 90:
        raise ValueError("A native output comparison requires a finite original deadline")
    if acquire:
        directory.mkdir()
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("The complete native output comparisons require a regular directory")
    interface = AigerInterface(frozenset(input_order), frozenset(output_order))
    intern: dict[tuple[int, int], int] = {}
    graphs = (
        _Aiger.parse(reference, interface, intern),
        _Aiger.parse(candidate, interface, intern),
    )
    groups = _groups(*graphs, output_order)
    artifacts: dict[str, str] = {}
    expected_directories = {directory}
    comparison_count = 0
    retained_bytes = 0
    cec = f"cec -T {max(1, min(60, int(timeout)))} -n -v reference.aig candidate.aig"

    def node(path: Path) -> None:
        """Register one deterministic proof stage with direct custody."""
        if acquire:
            path.mkdir()
        if path.is_symlink() or not path.is_dir():
            raise ValueError("A native output comparison stage must be a regular directory")
        expected_directories.add(path)

    def artifact(path: Path) -> None:
        """Bind one directly retained native artifact to the complete comparison."""
        nonlocal retained_bytes
        _regular(path)
        retained_bytes += path.stat().st_size
        if retained_bytes > _MAX_BYTES:
            raise ValueError("The complete native output comparisons exceed the retention budget")
        artifacts[path.relative_to(directory).as_posix()] = digest(path)

    def command(
        root: Path, label: str, script: str, limit: float, *, comparison: bool = False
    ) -> bool:
        """Validate the exact operating-system command and require an actual native verdict."""
        nonlocal comparison_count
        log = root / (label + ".log")
        argv = [str(tools.abc), "-c", script]
        if comparison:
            comparison_count += 1
            if comparison_count > _MAX_COMPARISONS:
                raise ValueError("The bounded native output comparison is incomplete")
        if acquire:
            try:
                tools.run(argv, root, log, timeout=limit)
            except SynthesisToolError as error:
                if not comparison or not error.timed_out or not _needs_further_proof(log, tools):
                    raise
        artifact(log)
        _check_log(log)
        artifact(root / (label + ".command.json"))
        validate_checkpoint_command(root, label, tools, [argv[0], "-s", *argv[1:]], limit)
        return not comparison or not _needs_further_proof(log, tools)

    def sign_cofactors(root: Path, selected: AigerInterface) -> None:
        """Compare disjoint original input regions of the last undecided output."""
        positions = _sign_positions(input_order)
        for role in _ROLES:
            command(root, role + "-order", f"read_aiger {role}.aig; print_io", min(timeout, 15))
        names = _native_inputs(root / "reference-order.log", len(input_order))
        if names != _native_inputs(root / "candidate-order.log", len(input_order)):
            raise ValueError("The native output comparison changes original by-order inputs")
        refinements: tuple[int, ...] = ()
        if len(positions) == 1:
            port, _, suffix = input_order[positions[0]].rpartition("[")
            high = int(suffix[:-1]) - 1
            if high >= 0:
                # Split at the highest byte boundary, then at the bit just below the sign.
                # Each cut fixes an original input bit and requires proof of both polarities.
                bits = dict.fromkeys((high - high % 8, high))
                by_name = {name: index for index, name in enumerate(input_order)}
                cuts = (f"{port}[{bit}]" for bit in bits)
                refinements = tuple(by_name[cut] for cut in cuts if cut in by_name)

        def compare_region(
            case: Path, assignments: tuple[tuple[int, int], ...], cuts: tuple[int, ...]
        ) -> None:
            """Prove one region or require positive evidence for both halves of its next cut."""
            node(case)
            restrictions = "".join(f"cof {names[index]} {value}; " for index, value in assignments)
            source = "../" * len(case.relative_to(root).parts)
            for role in _ROLES:
                script = (
                    f"read_aiger {source}{role}.aig; logic; {restrictions}"
                    f"strash; write_aiger -s {role}.aig; print_io"
                )
                command(case, role + "-cofactor", script, min(timeout, 60))
                path = case / (role + ".aig")
                artifact(path)
                check_aiger_header(path, selected)
                if _native_inputs(case / (role + "-cofactor.log"), len(input_order)) != names:
                    raise ValueError("A native output cofactor changed the original input order")
            if command(case, "equivalence", cec, timeout, comparison=True):
                return
            if not cuts:
                raise ValueError("An original input region has no complete native output proof")
            cut = cuts[0]
            bit = input_order[cut].rsplit("[", 1)[1][:-1]
            for value in (0, 1):
                compare_region(case / f"bit-{bit}-{value}", (*assignments, (cut, value)), cuts[1:])

        for branch in itertools.product((0, 1), repeat=len(positions)):
            case = root / ("sign-" + "".join(map(str, branch)))
            compare_region(case, tuple(zip(positions, branch, strict=True)), refinements)

    def compare(root: Path, chosen: tuple[int, ...]) -> None:
        """Compare a complete group or recursively check both of its disjoint output halves."""
        node(root)
        selected = AigerInterface(interface.inputs, frozenset(output_order[i] for i in chosen))
        for role, graph in zip(_ROLES, graphs, strict=True):
            variant = root / (role + ".aig")
            native = graph.select(chosen)
            if acquire:
                with variant.open("xb") as stream:
                    stream.write(native)
            artifact(variant)
            if variant.read_bytes() != native:
                raise ValueError("A native output group changed its original AIGER graph")
            check_aiger_header(variant, selected)
        # Once a group is subdivided, prove a two-output remainder directly by output.
        if (len(chosen) != 2 or root.parent == directory) and command(
            root, "equivalence", cec, timeout, comparison=True
        ):
            return
        if len(chosen) == 1:
            sign_cofactors(root, selected)
            return
        middle = len(chosen) // 2
        compare(root / "first", chosen[:middle])
        compare(root / "second", chosen[middle:])

    for index, group in enumerate(groups):
        compare(directory / f"group-{index:03d}", group)
    for path in directory.rglob("*"):
        if path.is_symlink() or (
            path not in expected_directories
            and path.relative_to(directory).as_posix() not in artifacts
        ):
            raise ValueError("The native output comparisons contain an unrecognized artifact")
    return {
        "input_bits": len(input_order),
        "output_bits": len(output_order),
        "representative_original_output_indices": [list(group) for group in groups],
        "native_comparisons": comparison_count,
        "artifact_sha256": dict(sorted(artifacts.items())),
    }
