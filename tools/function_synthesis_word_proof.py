# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete word-stage correspondence

"""Prove local product identities and every original word-function connection."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from tools.function_synthesis_netlist import Bit, Netlist
from tools.function_synthesis_polynomial import ExactWordPolynomials, WordProofWork
from tools.function_synthesis_word_graph import WordGraph, word_dependencies
from tools.function_synthesis_word_groups import (
    GroupedWordComposition,
    prove_grouped_word_composition,
)


def _write(path: Path, value: object) -> None:
    """Preserve one exact private derivation without replacing earlier evidence."""
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2) + "\n")


@dataclass(frozen=True, slots=True)
class WordComposition:
    """Retain the checked original graph and all universal product identities.

    Parameters
    ----------
    product_count : int
        Actual original multiplications proved over their complete result words.
    non_multiplier_count : int
        Original cells with identical functions and all corresponding edges.
    wire_bindings : tuple of (int, int or str)
        Complete checked reference-to-candidate wire correspondence.
    original_order : tuple of str
        Acyclic original function order from complete result dependencies.
    covered_product_cells : tuple of str
        Actual DSP and adder cells used by the universal product identities.

    """

    product_count: int
    non_multiplier_count: int
    wire_bindings: tuple[tuple[int, Bit], ...]
    original_order: tuple[str, ...]
    covered_product_cells: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return exact completed component counts without claiming whole mapped acceptance."""
        return {
            "products_proven": self.product_count,
            "unchanged_functions_checked": self.non_multiplier_count,
            "wire_bindings_checked": len(self.wire_bindings),
            "original_acyclic_order": list(self.original_order),
            "product_cells": list(self.covered_product_cells),
        }


def _original_order(module: Netlist, primary: Mapping[int, str]) -> tuple[str, ...]:
    """Order original result dependencies after complete global combinational validation."""
    drivers: dict[Bit, str] = {}
    for name, cell in module.cells.items():
        for port, direction in cell.directions.items():
            if direction != "output":
                continue
            for bit in cell.connections[port]:
                if isinstance(bit, str):
                    continue
                if bit in primary or (bit in drivers and drivers[bit] != name):
                    raise ValueError(
                        "The original word graph constrains an input or has multiple drivers"
                    )
                drivers[bit] = name
    done: set[str] = set()
    order: list[str] = []

    def visit_bit(bit: Bit) -> None:
        """Follow actual data dependencies and refuse unknown original values."""
        if bit in {"0", "1"} or bit in primary:
            return
        if isinstance(bit, str) or bit not in drivers:
            raise ValueError("The original word graph has an unknown or undriven data dependency")
        visit_cell(drivers[bit])

    def visit_cell(name: str) -> None:
        """Retain a complete topological order without assuming equal cut inputs."""
        if name in done:
            return
        cell = module.cells[name]
        for port, direction in cell.directions.items():
            if direction != "input":
                continue
            if cell.kind == "$memrd_v2" and port == "CLK":
                # The global scan required an asynchronous ROM; its clock pin is inactive.
                continue
            for bit in cell.connections[port]:
                visit_bit(bit)
        done.add(name)
        order.append(name)

    for bit in module.ports["result"].bits:
        visit_bit(bit)
    return tuple(order)


def prove_word_composition(
    original: Netlist,
    mapped: Netlist,
    input_widths: Mapping[str, int],
    output_width: int,
    directory: Path | None = None,
    *,
    deadline: float | None = None,
) -> WordComposition | GroupedWordComposition:
    """Prove every actual multiplication and all remaining original function edges.

    Parameters
    ----------
    original, mapped : Netlist
        Actual checkpoints before and after DSP mapping of the same complete unit.
    input_widths : mapping of str to int
        Every independently driven original argument port and full width.
    output_width : int
        Complete original result width.
    directory : pathlib.Path or None, optional
        New private directory retaining wire correspondence and exact derivations.
        None recomputes the bounded identities without writing retained artifacts.
    deadline : float or None, optional
        Inherited monotonic case deadline on both individual and grouped paths.
        Standalone work uses the same existing 3300-second shared allowance.

    Returns
    -------
    WordComposition or GroupedWordComposition
        Universal local identities plus the checked acyclic whole-function graph.

    Raises
    ------
    OSError, ValueError
        For unsupported words, changed functions, edges, memories or any nonzero
        exact product residue. No incomplete identity becomes an accepted component.

    """
    now = time.monotonic()
    if deadline is not None and (isinstance(deadline, bool) or not math.isfinite(deadline)):
        raise ValueError("A word proof requires a finite inherited deadline")
    work = WordProofWork(min(now + 3300, deadline) if deadline is not None else now + 3300)
    work.check()
    if any(
        cell.kind != "$mul" and name not in mapped.cells for name, cell in original.cells.items()
    ):
        return prove_grouped_word_composition(
            original, mapped, input_widths, output_width, directory, deadline=work.deadline
        )
    algebra = ExactWordPolynomials(observer=work.observe)
    source = WordGraph(original, algebra, input_widths, output_width, reference_multiplication=True)
    candidate = WordGraph(mapped, algebra, input_widths, output_width)
    word_dependencies(source)
    word_dependencies(candidate)
    bindings: dict[int, Bit] = {}

    def bind(left: Bit, right: Bit) -> None:
        """Check constants and actual aliases before admitting a correspondence edge."""
        left, right = source.find(left), candidate.find(right)
        if isinstance(left, str) or isinstance(right, str):
            if left != right:
                raise ValueError("An original constant or unknown word correspondence changed")
            return
        if left in bindings and bindings[left] != right:
            raise ValueError("An original wire has conflicting mapped correspondences")
        bindings[left] = right

    def corresponding(bit: Bit) -> Bit:
        """Require an already checked actual wire for each local product boundary."""
        bit = source.find(bit)
        if isinstance(bit, str):
            return bit
        if bit not in bindings:
            raise ValueError("An actual product operand or output lacks a checked correspondence")
        return bindings[bit]

    for name, module_port in original.ports.items():
        for left_bit, right_bit in zip(module_port.bits, mapped.ports[name].bits, strict=True):
            bind(left_bit, right_bit)
    for name in sorted(original.names.keys() & mapped.names.keys()):
        source_bits, candidate_bits = original.names[name], mapped.names[name]
        if (
            len(source_bits) != len(candidate_bits)
            or original.offsets[name] != mapped.offsets[name]
        ):
            raise ValueError("A shared named synthesis vector has different actual geometry")
        for source_bit, candidate_bit in zip(source_bits, candidate_bits, strict=True):
            bind(source_bit, candidate_bit)
    checked: set[str] = set()
    for name, cell in original.cells.items():
        if cell.kind == "$mul":
            continue
        target = mapped.cells.get(name)
        if target is None or (cell.kind, cell.parameters, cell.directions) != (
            target.kind,
            target.parameters,
            target.directions,
        ):
            raise ValueError("An original non-multiplier function changed during DSP mapping")
        for connection_name, bits in cell.connections.items():
            actual = target.connections[connection_name]
            if len(bits) != len(actual):
                raise ValueError("An original function has different mapped connection widths")
            for original_bit, mapped_bit in zip(bits, actual, strict=True):
                bind(original_bit, mapped_bit)
        checked.add(name)
    if original.memory_identity != mapped.memory_identity:
        raise ValueError("Original memory declarations changed during DSP mapping")
    if any(
        cell.kind not in {"$add", "$sub", "$mul", "DSP48E1"}
        for name, cell in mapped.cells.items()
        if name not in checked
    ):
        raise ValueError(
            "A mapped function has no arithmetic disposition or checked original correspondence"
        )
    order = _original_order(original, original.primary_inputs(input_widths, output_width))
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=False)
    products = 0
    covered: set[str] = set()
    for name, cell in original.cells.items():
        work.check()
        if cell.kind != "$mul":
            continue
        operands = {port: cell.connections[port] for port in ["A", "B"]}
        result = cell.connections["Y"]
        actual_operands = {
            port: tuple(corresponding(bit) for bit in bits) for port, bits in operands.items()
        }
        actual_result = tuple(corresponding(bit) for bit in result)
        retained = mapped.cells.get(name)
        retained_name = None
        if retained is not None and retained.kind == "$mul":
            if (
                retained.parameters != cell.parameters
                or retained.directions != cell.directions
                or any(
                    tuple(candidate.find(bit) for bit in retained.connections[port]) != bits
                    for port, bits in (*actual_operands.items(), ("Y", actual_result))
                )
            ):
                raise ValueError("A retained multiplication changed its parameters or actual ports")
            retained_name = name
        ring = ExactWordPolynomials(observer=work.observe)
        gold = WordGraph(
            original,
            ring,
            input_widths,
            output_width,
            operands=operands,
            result=result,
            reference_multiplication=True,
        )
        gate = WordGraph(
            mapped,
            ring,
            input_widths,
            output_width,
            operands=actual_operands,
            result=actual_result,
            retained_multiplication=retained_name,
        )
        reference_value, candidate_value = gold.read(result, False), gate.read(actual_result, False)
        difference = ring.residue(
            ring.add(reference_value, ring.scale(candidate_value, -1)), len(result)
        )
        if directory is not None:
            _write(
                directory / f"product-{products:03d}.json",
                {
                    "original_cell": name,
                    "complete_product_bits": len(result),
                    "original_operand_widths": {port: len(bits) for port, bits in operands.items()},
                    "reference": ring.describe(reference_value),
                    "mapped": ring.describe(candidate_value),
                    "difference": ring.describe(difference),
                    "reference_cells": sorted(gold.visited),
                    "mapped_cells": sorted(gate.visited),
                },
            )
        if difference:
            raise ValueError("A complete actual multiplication word is not equivalent")
        covered.update(gate.visited)
        products += 1
    proof = WordComposition(
        products, len(checked), tuple(sorted(bindings.items())), order, tuple(sorted(covered))
    )
    work.check()
    if directory is not None:
        _write(directory / "wire-bindings.json", proof.wire_bindings)
        _write(directory / "word-composition.json", proof.to_dict())
    work.check()
    return proof
