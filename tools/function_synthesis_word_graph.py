# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual configured DSP word semantics

"""Interpret checked combinational word cones over their actual connected bits."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

from tools.function_synthesis_dsp_ports import MacroConfiguration as MacroConfiguration
from tools.function_synthesis_dsp_ports import configured_dsp as configured_dsp
from tools.function_synthesis_netlist import Bit, Netlist, parameter_integer
from tools.function_synthesis_polynomial import ExactWordPolynomials, Polynomial

Driver: TypeAlias = tuple[str, str, int, int]


class WordGraph:
    """Read exact finite words without assuming equal upstream operands.

    Parameters
    ----------
    module : Netlist
        Actual original or post-DSP function module.
    algebra : ExactWordPolynomials
        Shared exact integer algebra for one universal local identity.
    input_widths : mapping of str to int
        Complete independently driven original argument ports, including binary inputs.
    output_width : int
        Complete original result width.
    operands : mapping of str to tuple of int or str, optional
        Actual local operand wires made universally free for this identity only.
        Their upstream equality must be established by whole graph correspondence.
    result : tuple of int or str, optional
        Actual product result; an operand boundary must never cross these wires.
    reference_multiplication : bool, optional
        Whether actual original $mul cells may supply reference product words.

    """

    def __init__(
        self,
        module: Netlist,
        algebra: ExactWordPolynomials,
        input_widths: Mapping[str, int],
        output_width: int,
        *,
        operands: Mapping[str, tuple[Bit, ...]] | None = None,
        result: tuple[Bit, ...] = (),
        reference_multiplication: bool = False,
    ) -> None:
        """Validate the complete interface, aliases, drivers and local boundaries."""
        self.module, self.algebra = module, algebra
        self.reference_multiplication = reference_multiplication
        self.parents: dict[Bit, Bit] = {}
        self.cells = module.cells
        primary = module.primary_inputs(input_widths, output_width)
        self.visited: set[str] = set()
        self.stack: set[str] = set()
        self.memo: dict[tuple[str, str], Polynomial] = {}
        self.drivers: dict[Bit, Driver] = {}
        for cell in self.cells.values():
            if cell.kind != "DSP48E1":
                continue
            _, a_input, b_input = configured_dsp(cell)
            connections = cell.connections
            aliases = {
                "ACOUT": connections["ACIN" if a_input == "CASCADE" else "A"],
                "BCOUT": connections["BCIN" if b_input == "CASCADE" else "B"],
                "PCOUT": connections["P"],
            }
            for port, values in aliases.items():
                if port not in connections:
                    continue
                for output, value in zip(connections[port], values, strict=True):
                    if isinstance(output, str):
                        continue
                    if output in primary and output != value:
                        raise ValueError(
                            "A DSP cascade output constrains an original primary input"
                        )
                    self.union(output, value)
        resolved = {self.find(bit): name for bit, name in primary.items()}
        if len(resolved) != len(primary) or any(isinstance(bit, str) for bit in resolved):
            raise ValueError("Original independent primary input bits were constrained")
        self.leaves = {bit: algebra.boolean(name) for bit, name in resolved.items()}
        for name, cell in self.cells.items():
            for port, direction in cell.directions.items():
                if direction != "output" or (
                    cell.kind == "DSP48E1" and port in {"ACOUT", "BCOUT", "PCOUT"}
                ):
                    continue
                values = cell.connections[port]
                for index, bit in enumerate(values):
                    if isinstance(bit, str):
                        continue
                    key = self.find(bit)
                    if isinstance(key, str):
                        raise ValueError("A word function drives an aliased constant or unknown")
                    driver = (name, port, index, len(values))
                    if key in self.leaves:
                        raise ValueError("A function cell drives an original primary input")
                    if key in self.drivers and self.drivers[key] != driver:
                        raise ValueError("A word proof contains multiple non-alias drivers")
                    self.drivers[key] = driver
        output_bits = {self.find(bit) for bit in result if isinstance(bit, int)}
        for name, bits in (operands or {}).items():
            if not bits:
                raise ValueError("A local operand boundary must not be empty")
            for index, raw in enumerate(bits):
                bit = self.find(raw)
                if bit in {"0", "1"}:
                    continue
                if isinstance(bit, str) or bit in output_bits:
                    raise ValueError("A local operand contains an unknown or crosses its result")
                if bit not in self.leaves:
                    if bit not in self.drivers:
                        raise ValueError("A local operand boundary is undriven")
                    self.leaves[bit] = algebra.boolean(f"local/{name}[{index}]")

    def find(self, bit: Bit) -> Bit:
        """Resolve checked cascade aliases without changing the represented value."""
        chain: list[Bit] = []
        while bit in self.parents:
            chain.append(bit)
            bit = self.parents[bit]
        for child in chain:
            self.parents[child] = bit
        return bit

    def union(self, left: Bit, right: Bit) -> None:
        """Identify two wires only through a qualified actual cascade connection."""
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if isinstance(a, str) and isinstance(b, str):
            raise ValueError("A DSP cascade has conflicting constant aliases")
        if isinstance(b, str):
            self.parents[a] = b
        else:
            self.parents[b] = a

    def _direct(self, bits: tuple[Bit, ...], signed: bool) -> Polynomial:
        """Read a word fragment consisting entirely of constants or universal bits."""
        value: Polynomial = {}
        for index, bit in enumerate(bits):
            term = self.algebra.constant(int(bit)) if bit in {"0", "1"} else self.leaves[bit]
            coefficient = (1 << index) - (
                (1 << len(bits)) if signed and index == len(bits) - 1 else 0
            )
            value = self.algebra.add(value, self.algebra.scale(term, coefficient))
        return value

    def read(self, raw: tuple[Bit, ...], signed: bool) -> Polynomial:
        """Interpret every actual bit in a finite connected word or signed slice."""
        bits = tuple(self.find(bit) for bit in raw)
        if not 1 <= len(bits) <= 64 or any(
            isinstance(bit, str) and bit not in {"0", "1"} for bit in bits
        ):
            raise ValueError("An exact product word has an unsupported width or unknown bit")
        result: Polynomial = {}
        index = 0
        while index < len(bits):
            bit = bits[index]
            if bit in {"0", "1"} or bit in self.leaves:
                end = index + 1
                while end < len(bits) and (bits[end] in {"0", "1"} or bits[end] in self.leaves):
                    end += 1
                value = self._direct(bits[index:end], signed and end == len(bits))
            else:
                if bit not in self.drivers:
                    raise ValueError("An actual product word has an undriven internal bit")
                name, port, low, width = self.drivers[bit]
                end, high = index + 1, low
                while end < len(bits) and self.drivers.get(bits[end]) == (
                    name,
                    port,
                    high + 1,
                    width,
                ):
                    high += 1
                    end += 1
                if signed:
                    extended = end
                    while extended < len(bits) and bits[extended] == bits[end - 1]:
                        extended += 1
                    if extended == len(bits):
                        end = extended
                value = self.algebra.slice(
                    self.raw(name, port), width, low, high, signed and end == len(bits)
                )
            result = self.algebra.add(result, self.algebra.scale(value, 1 << index))
            index = end
        return result

    def raw(self, name: str, port: str) -> Polynomial:
        """Interpret one qualified primitive or original arithmetic result exactly."""
        identity = (name, port)
        if identity in self.memo:
            return self.memo[identity]
        if name in self.stack:
            raise ValueError("An exact word proof contains a cyclic product cone")
        self.stack.add(name)
        self.visited.add(name)
        cell = self.cells[name]
        connections, parameters = cell.connections, cell.parameters
        if cell.kind == "DSP48E1" and port == "P":
            mode, a_input, b_input = configured_dsp(cell)
            left_bits = connections["ACIN" if a_input == "CASCADE" else "A"][:25]
            right_bits = connections["BCIN" if b_input == "CASCADE" else "B"]
            product = self.algebra.multiply(self.read(left_bits, True), self.read(right_bits, True))
            if mode == "0000101":
                value = product
            elif mode == "0010101":
                value = self.algebra.add(product, self.read(connections["PCIN"], True))
            elif mode == "1010101":
                value = self.algebra.add(product, self.read(connections["PCIN"][17:], True))
            else:
                value = self.algebra.add(product, self.read(connections["C"], True))
        elif cell.kind in {"$add", "$sub", "$mul"} and port == "Y":
            if cell.kind == "$mul" and not self.reference_multiplication:
                raise ValueError("An unmapped multiplication remains in a DSP product cone")
            if any(
                parameter_integer(parameters, pin + "_WIDTH") != len(connections[pin])
                for pin in ["A", "B", "Y"]
            ):
                raise ValueError(
                    "An actual arithmetic cell has different parameter and port widths"
                )
            a_signed = parameter_integer(parameters, "A_SIGNED")
            b_signed = parameter_integer(parameters, "B_SIGNED")
            if a_signed not in {0, 1} or b_signed not in {0, 1}:
                raise ValueError("An arithmetic signedness flag is not a single Boolean value")
            if cell.kind == "$mul" and a_signed != b_signed:
                raise ValueError("A mixed-signedness multiplication has no qualified word contract")
            signed = bool(a_signed and b_signed)
            left, right = self.read(connections["A"], signed), self.read(connections["B"], signed)
            value = (
                self.algebra.multiply(left, right)
                if cell.kind == "$mul"
                else self.algebra.add(
                    left, self.algebra.scale(right, -1 if cell.kind == "$sub" else 1)
                )
            )
        else:
            raise ValueError("An actual product cone uses an unqualified word operation")
        self.stack.remove(name)
        self.memo[identity] = value
        return value
