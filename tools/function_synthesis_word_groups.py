# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete grouped arithmetic word identities

"""Prove structural arithmetic components without assuming removed product cuts."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from tools.function_synthesis_netlist import Bit, Netlist, object_mapping
from tools.function_synthesis_polynomial import ExactWordPolynomials, WordProofWork
from tools.function_synthesis_word_graph import (
    WordGraph,
    active_word_inputs,
    word_cone,
    word_dependencies,
    word_topological_order,
)

_ARITHMETIC = frozenset({"$add", "$sub", "$mul"})


@dataclass(frozen=True, slots=True)
class GroupedWordComposition:
    """Keep joint identities distinct from individual product proofs.

    Parameters
    ----------
    record : dict of str to object
        Deterministic complete graph, frontier, root and observability certificate.
    work : WordProofWork
        Actual measured work; never used as a deterministic identity field.
    """

    record: dict[str, object]
    work: WordProofWork

    def to_dict(self) -> dict[str, object]:
        """Return the complete joint certificate without inventing per-product equality."""
        return object_mapping(json.loads(json.dumps(self.record)))


def prove_grouped_word_composition(
    original: Netlist,
    mapped: Netlist,
    input_widths: Mapping[str, int],
    output_width: int,
    directory: Path | None = None,
    *,
    deadline: float | None = None,
) -> GroupedWordComposition:
    """Prove full arithmetic components and inductively check every outside function.

    Parameters
    ----------
    original, mapped : Netlist
        Actual complete checkpoints, with immutable external raw custody bindings.
    input_widths : mapping of str to int
        Every independent original primary input and its complete width.
    output_width : int
        Complete original result width.
    directory : pathlib.Path or None, optional
        New retained derivations or a nonzero-residue refusal with a local
        Boolean witness when available; None performs read-only recomputation.
    deadline : float or None, optional
        Inherited absolute monotonic case deadline. No invocation extends it;
        standalone work is capped at the existing 3300-second allowance.

    Notes
    -----
    Both complete graphs pass ``word_dependencies`` before grouping. Its global
    state refusal includes registers, writes and clocked memory reads outside
    every result cone. Only under that checked combinational contract are
    primary outputs the complete global observability sinks.

    Surviving functions bind every actual pin before induction. The collapsed
    source DAG therefore establishes their inputs without a second copy of the
    binding table. A group's frontier contains every outside defining wire;
    making exactly those wires leaves keeps its source traversal inside the
    group. Fixed-point mapped cones are disjoint and supply global accounting
    independently of polynomial traversal. Output ports are checked against
    these proved anchors after that accounting; a port supplies an anchor only
    where no shared name or surviving pin already supplies one.
    ``Netlist.read`` already ties each named port to its declared connections,
    so the final port comparison also validates its shared output name.

    Returns
    -------
    GroupedWordComposition
        Joint complete-word identities, structural induction and global accounting.

    Raises
    ------
    OSError, ValueError
        For incomplete, changed, stateful, cyclic or non-equivalent graphs, or
        explicit expression-size or shared-wall budget exhaustion.
    """
    now = time.monotonic()
    if deadline is not None and (isinstance(deadline, bool) or not math.isfinite(deadline)):
        raise ValueError("A grouped word proof requires a finite inherited deadline")
    work = WordProofWork(min(now + 3300, deadline) if deadline is not None else now + 3300)
    work.check()
    original, mapped = (
        replace(module, cells=dict(sorted(module.cells.items()))) for module in (original, mapped)
    )
    algebra = ExactWordPolynomials(observer=work.observe)
    source = WordGraph(original, algebra, input_widths, output_width, reference_multiplication=True)
    candidate = WordGraph(mapped, algebra, input_widths, output_width)
    primary_names = {
        source.find(bit): name
        for bit, name in original.primary_inputs(input_widths, output_width).items()
    }
    source_dependencies, mapped_dependencies = (
        word_dependencies(source),
        word_dependencies(candidate),
    )
    if original.memory_identity != mapped.memory_identity:
        raise ValueError("Original memory declarations changed during grouped DSP mapping")
    bindings: dict[int, int] = {}

    def bind(left: Bit, right: Bit) -> None:
        """Retain a proposed wire anchor without treating it as a proved equality."""
        left, right = source.find(left), candidate.find(right)
        if isinstance(left, str) or isinstance(right, str):
            if left != right:
                raise ValueError("A grouped constant or unknown correspondence changed")
        elif left in bindings and bindings[left] != right:
            raise ValueError("A grouped wire has conflicting mapped correspondences")
        else:
            bindings[left] = right

    def corresponding(raw: Bit) -> Bit:
        """Require a complete mapped anchor for every boundary bit."""
        bit = source.find(raw)
        if bit in {"0", "1"}:
            return bit
        if isinstance(bit, str) or bit not in bindings:
            raise ValueError("A grouped operand or complete root lacks a correspondence")
        return bindings[bit]

    for name, module_port in original.ports.items():
        if module_port.direction == "input":
            for left_bit, right_bit in zip(module_port.bits, mapped.ports[name].bits, strict=True):
                bind(left_bit, right_bit)
    for name in sorted(original.names.keys() & mapped.names.keys()):
        if (len(original.names[name]), original.offsets[name]) != (
            len(mapped.names[name]),
            mapped.offsets[name],
        ):
            raise ValueError("A shared grouped vector has different actual geometry")
        if name not in original.ports or original.ports[name].direction != "output":
            for left_bit, right_bit in zip(original.names[name], mapped.names[name], strict=True):
                bind(left_bit, right_bit)
    missing = set(original.cells) - mapped.cells.keys()
    if not missing or any(original.cells[name].kind not in _ARITHMETIC for name in missing):
        raise ValueError("A grouped proof requires disappeared arithmetic functions only")
    for name in sorted(original.cells.keys() & mapped.cells.keys()):
        left, right = original.cells[name], mapped.cells[name]
        if (left.kind, left.parameters, left.directions) != (
            right.kind,
            right.parameters,
            right.directions,
        ):
            raise ValueError("An original surviving function changed during grouped mapping")
        for port, bits in left.connections.items():
            for a, b in zip(bits, right.connections[port], strict=True):
                bind(a, b)
    for name, module_port in original.ports.items():
        if module_port.direction == "output":
            for a, b in zip(module_port.bits, mapped.ports[name].bits, strict=True):
                if source.find(a) not in bindings:
                    bind(a, b)
    arithmetic = {name for name, cell in original.cells.items() if cell.kind in _ARITHMETIC}
    neighbours = {name: set(source_dependencies[name]) & arithmetic for name in arithmetic}
    for name in arithmetic:
        for parent in tuple(neighbours[name]):
            neighbours[parent].add(name)
    groups: list[set[str]] = []
    remaining = set(missing)
    while remaining:
        members: set[str] = set()
        pending = {min(remaining)}
        while pending:
            name = pending.pop()
            members.add(name)
            pending.update(neighbours[name] - members)
        if any(original.cells[name].kind != "$mul" for name in members & missing):
            groups.append(members)
        else:
            groups.extend({name} for name in sorted(members & missing))
        remaining -= members

    def boundary(members: set[str]) -> tuple[list[tuple[str, str]], tuple[int, ...]]:
        """Keep every complete externally consumed result and actual outside driver bit."""
        uses = {
            source.find(bit)
            for name, cell in original.cells.items()
            if name not in members
            for bit in active_word_inputs(cell)
        } | {
            source.find(bit)
            for p in original.ports.values()
            if p.direction == "output"
            for bit in p.bits
        }
        roots = [
            (name, port)
            for name in sorted(members)
            for port in sorted(original.cells[name].connections)
            if original.cells[name].directions[port] == "output"
            and any(source.find(bit) in uses for bit in original.cells[name].connections[port])
        ]
        frontier = {
            source.find(bit)
            for name in members
            for bit in active_word_inputs(original.cells[name])
            if isinstance(source.find(bit), int)
            and (
                source.find(bit) not in source.drivers
                or source.drivers[source.find(bit)][0] not in members
            )
        }
        if not roots:
            raise ValueError("An arithmetic group lacks a complete observable boundary")
        return roots, tuple(
            sorted(
                (bit for bit in frontier if isinstance(bit, int)),
                key=lambda bit: source.drivers.get(bit, ("", primary_names.get(bit, ""), 0, 1)),
            )
        )

    while True:
        work.check()
        boundaries = [boundary(members) for members in groups]
        cones = [
            word_cone(
                candidate,
                tuple(
                    corresponding(bit) for n, p in roots for bit in original.cells[n].connections[p]
                ),
                {corresponding(bit) for bit in frontier},
            )
            for roots, frontier in boundaries
        ]
        overlap = next(
            (
                (i, j)
                for i in range(len(groups))
                for j in range(i + 1, len(groups))
                if cones[i] & cones[j]
            ),
            None,
        )
        if overlap is None:
            break
        i, j = overlap
        groups[i] |= groups.pop(j)
    plans = sorted(
        zip(groups, boundaries, cones, strict=True), key=lambda plan: tuple(sorted(plan[0]))
    )
    grouped = {
        name: f"group/{index:04d}"
        for index, (members, _, _) in enumerate(plans)
        for name in members
    }
    unit = {name: grouped.get(name, "cell/" + name) for name in original.cells}
    dependencies: dict[str, set[str]] = {value: set() for value in unit.values()}
    for name, parents in source_dependencies.items():
        dependencies[unit[name]].update(
            unit[parent] for parent in parents if unit[parent] != unit[name]
        )
    ordered_units = word_topological_order(dependencies)
    records: list[dict[str, object]] = []
    covered: set[str] = set()
    for identity in ordered_units:
        work.check()
        if identity.startswith("cell/"):
            continue
        members, (roots, frontier), mapped_members = plans[int(identity.split("/")[1])]
        actual_frontier = tuple(corresponding(bit) for bit in frontier)
        if len(set(actual_frontier)) != len(frontier):
            raise ValueError("Independent grouped frontier wires were aliased or forced constant")
        operands = {f"frontier{index:05d}": (bit,) for index, bit in enumerate(frontier)}
        actual_operands = {
            name: tuple(corresponding(bit) for bit in bits) for name, bits in operands.items()
        }
        all_roots = tuple(bit for n, p in roots for bit in original.cells[n].connections[p])
        actual_roots = tuple(corresponding(bit) for bit in all_roots)
        ring = ExactWordPolynomials(observer=work.observe)
        gold = WordGraph(
            original,
            ring,
            input_widths,
            output_width,
            operands=operands,
            result=all_roots,
            reference_multiplication=True,
        )
        gate = WordGraph(
            mapped, ring, input_widths, output_width, operands=actual_operands, result=actual_roots
        )
        root_records: list[dict[str, object]] = []
        for name, port in roots:
            bits = original.cells[name].connections[port]
            actual = tuple(corresponding(bit) for bit in bits)
            reference_value, candidate_value = gold.read(bits, False), gate.read(actual, False)
            difference = ring.residue(
                ring.add(reference_value, ring.scale(candidate_value, -1)), len(bits)
            )
            if difference:
                if directory is not None:
                    directory.mkdir(parents=True, exist_ok=False)
                    refusal = {
                        "schema_version": "sc-neurocore.grouped-word-refusal.v1",
                        "status": "equality-not-established",
                        "source_members": sorted(members),
                        "cell": name,
                        "port": port,
                        "width": len(bits),
                        "source_bits": list(bits),
                        "mapped_bits": list(actual),
                        "frontier": [
                            [a, b] for a, b in zip(frontier, actual_frontier, strict=True)
                        ],
                        "reference": ring.describe(reference_value),
                        "mapped": ring.describe(candidate_value),
                        "difference": ring.describe(difference),
                        "witness": ring.boolean_witness(difference),
                        "witness_scope": "independent-local-frontier-not-primary-reachability",
                    }
                    with (directory / "group-refusal.json").open("x") as stream:
                        stream.write(json.dumps(refusal, sort_keys=True, indent=2) + "\n")
                raise ValueError("Complete grouped arithmetic word equality was not established")
            root_records.append(
                {
                    "cell": name,
                    "port": port,
                    "source_bits": list(bits),
                    "mapped_bits": list(actual),
                    "width": len(bits),
                    "ring": f"Z/(2^{len(bits)}Z)",
                    "reference": ring.describe(reference_value),
                    "mapped": ring.describe(candidate_value),
                    "difference": [],
                }
            )
        covered.update(mapped_members)
        records.append(
            {
                "schema_version": "sc-neurocore.grouped-word-equivalence.v1",
                "source_members": sorted(members),
                "mapped_members": sorted(mapped_members),
                "frontier": [[a, b] for a, b in zip(frontier, actual_frontier, strict=True)],
                "source_arithmetic": {
                    name: {
                        "operator": original.cells[name].kind,
                        "parameters": dict(sorted(original.cells[name].parameters.items())),
                    }
                    for name in sorted(members)
                },
                "mapped_arithmetic": {
                    name: {
                        "operator": mapped.cells[name].kind,
                        "parameters": dict(sorted(mapped.cells[name].parameters.items())),
                    }
                    for name in sorted(mapped_members)
                },
                "roots": root_records,
                "atoms": len(ring.atom_keys),
                "expression_term_limit": ring.maximum_terms,
            }
        )
    unchanged = set(original.cells) - grouped.keys()
    extra = set(mapped.cells) - unchanged - covered
    sinks = tuple(
        bit for port in mapped.ports.values() if port.direction == "output" for bit in port.bits
    )
    if extra & word_cone(candidate, sinks, set()):
        raise ValueError("Unaccounted mapped arithmetic reaches an original output")
    for name, module_port in original.ports.items():
        if module_port.direction == "output":
            for a, b in zip(module_port.bits, mapped.ports[name].bits, strict=True):
                if corresponding(a) != candidate.find(b):
                    raise ValueError("An original output lacks a complete proved grouped driver")
    if any(mapped.cells[name].kind not in _ARITHMETIC | {"DSP48E1"} for name in extra):
        raise ValueError("A globally unobserved mapped function has no arithmetic disposition")
    work.check()
    proof = GroupedWordComposition(
        {
            "schema_version": "sc-neurocore.word-composition.v2",
            "products_proven": 0,
            "products_covered_jointly": sum(original.cells[n].kind == "$mul" for n in grouped),
            "unchanged_functions_checked": len(unchanged),
            "wire_bindings": [[a, b] for a, b in sorted(bindings.items())],
            "original_acyclic_order": list(word_topological_order(source_dependencies)),
            "mapped_acyclic_order": list(word_topological_order(mapped_dependencies)),
            "composition_order": list(ordered_units),
            "arithmetic_groups": records,
            "globally_unobserved_mapped_cells": sorted(extra),
            "global_output_bits": list(sinks),
            "post_dsp_cell_count": len(mapped.cells),
            "semantic_graph_sha256": [
                hashlib.sha256(
                    json.dumps(module.raw, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                for module in (original, mapped)
            ],
        },
        work,
    )
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=False)
        for name, payload in [
            ("word-composition.json", proof.to_dict()),
            ("group-work.json", work.to_dict()),
        ]:
            with (directory / name).open("x") as stream:
                stream.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    work.check()
    return proof
