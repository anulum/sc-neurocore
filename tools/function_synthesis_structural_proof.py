# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Exact original function graph comparison

"""Compare complete original output functions using exact canonical bit expressions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from tools.function_synthesis_netlist import Bit, Netlist, NetlistCell, parameter_integer

_ARITHMETIC = frozenset({"$add", "$sub", "$mul", "$neg"})
_BINARY = frozenset({"$add", "$sub", "$mul", "$div", "$lt", "$le", "$eq", "$gt", "$sshl", "$sshr"})
_UNARY = frozenset({"$neg", "$logic_not", "$reduce_bool"})


class UnsupportedStructuralOperation(ValueError):
    """Identify an operation requiring a separate complete native proof."""

    def __init__(self, operation: str) -> None:
        """Retain the actual unqualified operation without treating it as equivalent."""
        super().__init__("An original operation has no supported exact structural interpretation")
        self.operation = operation


@dataclass(frozen=True, slots=True)
class StructuralComparison:
    """Retain complete output-root identities from an exact expression comparison.

    Parameters
    ----------
    matches : bool
        True only when every original output has exactly the same function key.
        False is structural mismatch, not a mathematical counterexample.
    node_count : int
        Number of exact canonical expressions interned in the shared table.
    original_roots, mapped_roots : tuple of int
        Function identities of all original output bits, in their actual order.

    """

    matches: bool
    node_count: int
    original_roots: tuple[int, ...]
    mapped_roots: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        """Describe the actual structural result without inventing a solver verdict."""
        return {
            "all_output_function_keys_identical": self.matches,
            "canonical_nodes": self.node_count,
            "original_output_roots": list(self.original_roots),
            "mapped_output_roots": list(self.mapped_roots),
        }


def _shape(cell: NetlistCell) -> None:
    """Check the actual geometry of each supported combinational word operation.

    Parameters
    ----------
    cell : NetlistCell
        Actual cell parameters, port directions and connected bit vectors.

    Raises
    ------
    ValueError
        For malformed geometry or unqualified multiplexer parameters.
    UnsupportedStructuralOperation
        If the operation needs a separate complete native proof.

    """
    if cell.kind == "$mux":
        expected = {"A": "input", "B": "input", "S": "input", "Y": "output"}
        connections = cell.connections
        if (
            cell.directions != expected
            or set(cell.parameters) != {"WIDTH"}
            or not (
                len(connections["S"]) == 1
                and len(connections["A"]) == len(connections["B"]) == len(connections["Y"])
            )
            or parameter_integer(cell.parameters, "WIDTH") != len(connections["Y"])
        ):
            raise ValueError("An original multiplexer has invalid parameter or port geometry")
    elif cell.kind in _BINARY | _UNARY:
        expected = {"A": "input", "Y": "output"}
        if cell.kind in _BINARY:
            expected["B"] = "input"
        if cell.directions != expected or any(
            parameter_integer(cell.parameters, port + "_WIDTH") != len(bits)
            for port, bits in cell.connections.items()
        ):
            raise ValueError("An original word function has invalid parameter or port geometry")
    else:
        raise UnsupportedStructuralOperation(cell.kind)


class _Expressions:
    """Intern complete immutable function keys and retain their actual dependencies."""

    def __init__(self) -> None:
        """Create an empty exact expression table with a finite node budget."""
        self.keys: dict[tuple[object, ...], int] = {}
        self.definitions: list[tuple[object, ...]] = []
        self.children: list[tuple[int, ...]] = []
        self.unknowns: set[int] = set()

    def intern(self, key: tuple[object, ...], children: tuple[int, ...] = ()) -> int:
        """Use full tuple equality, never a digest alone, to identify a function."""
        if key not in self.keys:
            if len(self.keys) >= 200_000:
                raise ValueError("The exact structural function node budget was exceeded")
            self.keys[key] = len(self.definitions)
            self.definitions.append(key)
            self.children.append(children)
        return self.keys[key]

    def outputs(
        self,
        module: Netlist,
        input_widths: Mapping[str, int],
        output_width: int,
        phase: str,
    ) -> tuple[int, ...]:
        """Trace all actual output functions from their complete original primary domain."""
        leaves = {
            bit: self.intern(("primary", name))
            for bit, name in module.primary_inputs(input_widths, output_width).items()
        }
        zero, one = self.intern(("constant", "0")), self.intern(("constant", "1"))
        drivers: dict[int, tuple[str, int]] = {}
        for name, cell in module.cells.items():
            for port, direction in cell.directions.items():
                if direction != "output":
                    continue
                if port != "Y":
                    raise ValueError(
                        "An original structural function has an unqualified output port"
                    )
                for index, raw in enumerate(cell.connections[port]):
                    if isinstance(raw, str):
                        continue
                    if raw in leaves or (raw in drivers and drivers[raw] != (name, index)):
                        raise ValueError(
                            "An original function has multiple drivers or constrains an input"
                        )
                    drivers[raw] = (name, index)
        memo: dict[int, int] = {}
        active: set[int] = set()

        def bit(raw: Bit) -> int:
            """Read an actual source bit and preserve every reachable semantic dependency."""
            if raw == "0":
                return zero
            if raw == "1":
                return one
            if isinstance(raw, str):
                unknown = self.intern(("unknown", phase, raw))
                self.unknowns.add(unknown)
                return unknown
            if raw in leaves:
                return leaves[raw]
            if raw in memo:
                return memo[raw]
            if raw in active or raw not in drivers:
                raise ValueError("An original structural function has a cycle or undriven bit")
            active.add(raw)
            name, index = drivers[raw]
            cell = module.cells[name]
            _shape(cell)
            parameters = tuple(sorted(cell.parameters.items()))
            if cell.kind == "$mux":
                a, b, s = (
                    bit(cell.connections[port][position])
                    for port, position in [
                        ("A", index),
                        ("B", index),
                        ("S", 0),
                    ]
                )
                if a == b or s == zero:
                    value = a
                elif s == one:
                    value = b
                else:
                    # Geometry was validated above. A scalar mux depends on
                    # its selected bits, not their lane or unused bus width.
                    value = self.intern(
                        ("mux-bit", a, b, s),
                        (a, b, s),
                    )
            else:
                arguments: list[tuple[str, tuple[int, ...]]] = []
                children: list[int] = []
                for port, direction in sorted(cell.directions.items()):
                    if direction != "input":
                        continue
                    wires = cell.connections[port]
                    if cell.kind in _ARITHMETIC:
                        wires = wires[: index + 1]
                    values = tuple(bit(wire) for wire in wires)
                    arguments.append((port, values))
                    children.extend(values)
                value = self.intern(
                    ("cell-bit", cell.kind, parameters, index, tuple(arguments)),
                    tuple(children),
                )
            active.remove(raw)
            memo[raw] = value
            return value

        roots = tuple(bit(raw) for raw in module.ports["result"].bits)
        reachable: set[int] = set()
        pending = list(roots)
        while pending:
            node = pending.pop()
            if node in reachable:
                continue
            reachable.add(node)
            pending.extend(self.children[node])
        if reachable & self.unknowns:
            raise ValueError("An original output function depends on unknown values")
        return roots


def compare_word_structure(
    original: Netlist,
    mapped: Netlist,
    input_widths: Mapping[str, int],
    output_width: int,
    directory: Path | None = None,
) -> StructuralComparison:
    """Compare every complete original output using exact bit-function identities.

    Parameters
    ----------
    original, mapped : Netlist
        Normalized actual original and reference-checkpoint modules.
    input_widths : mapping of str to int
        Every independent original argument port and its complete width.
    output_width : int
        Complete original result width; upper bits are never discarded.
    directory : pathlib.Path or None, optional
        New private directory retaining both root lists and full function keys.
        None computes the bounded comparison without writing artifacts.

    Returns
    -------
    StructuralComparison
        Exact equality or structural mismatch. A mismatch requires a separate
        native complete-domain proof; it does not prove different behavior.

    Raises
    ------
    OSError, ValueError
        For unsupported or malformed operations, changed primary geometry,
        cycles, unknown output dependencies or exceeded finite expression budget.

    """
    expressions = _Expressions()
    left = expressions.outputs(original, input_widths, output_width, "original")
    right = expressions.outputs(mapped, input_widths, output_width, "mapped")
    result = StructuralComparison(left == right, len(expressions.definitions), left, right)
    if directory is None:
        return result
    directory.mkdir(parents=True, exist_ok=False)
    for name, payload in [
        ("structural-comparison.json", result.to_dict()),
        ("structural-expressions.json", expressions.definitions),
    ]:
        with (directory / name).open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    return result
