# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — UVM generator RTL contracts and module-header parser

"""Represent and parse the RTL module surface consumed by UVM generation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class PortDirection(Enum):
    """SystemVerilog port direction tokens accepted by the UVM generator."""

    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"


class PortType(Enum):
    """SystemVerilog net/data type tokens emitted for module ports."""

    LOGIC = "logic"
    WIRE = "wire"
    REG = "reg"


@dataclass
class ModulePort:
    """Parsed RTL port metadata used by generated UVM components."""

    name: str
    direction: PortDirection
    port_type: PortType = PortType.LOGIC
    width: int = 1
    is_signed: bool = False
    is_array: bool = False
    array_size: int = 0

    @property
    def sv_decl(self) -> str:
        """Return the SystemVerilog declaration for this parsed port."""
        signed = " signed" if self.is_signed else ""
        width = f" [{self.width - 1}:0]" if self.width > 1 else ""
        arr = f" [0:{self.array_size - 1}]" if self.is_array else ""
        return f"{self.direction.value} {self.port_type.value}{signed}{width} {self.name}{arr}"

    @property
    def is_clock(self) -> bool:
        """Return whether the port name matches a supported clock convention."""
        return self.name.lower() in ("clk", "clock", "i_clk")

    @property
    def is_reset(self) -> bool:
        """Return whether the port name matches a supported reset convention."""
        return self.name.lower() in ("rst_n", "reset_n", "rst", "reset", "i_rst_n")


@dataclass
class ModuleParam:
    """RTL module parameter."""

    name: str
    value: str
    param_type: str = "int"


@dataclass
class RTLModule:
    """Parsed RTL module specification."""

    name: str
    ports: List[ModulePort]
    params: List[ModuleParam] = field(default_factory=list)
    is_sc_module: bool = True

    @classmethod
    def from_verilog_source(cls, source: str) -> RTLModule:
        """Parse a Verilog/SystemVerilog module header."""
        name_match = re.search(r"module\s+(\w+)", source)
        if not name_match:
            raise ValueError("No module declaration found")
        name = name_match.group(1)

        params = []
        param_block = re.search(r"#\s*\((.*?)\)\s*\(", source, re.DOTALL)
        if param_block:
            for match in re.finditer(
                r"parameter\s+(?:(\w+)\s+)?(\w+)\s*=\s*(\S+)", param_block.group(1)
            ):
                param_type = match.group(1) or "int"
                params.append(ModuleParam(match.group(2), match.group(3), param_type))

        ports = []
        if param_block:
            # The parameter regex consumes the opening parenthesis of the port list.
            rest = source[param_block.end() :]
            port_section = re.search(r"(.*?)\)\s*;", rest, re.DOTALL)
        else:
            port_section = re.search(r"\(\s*(.*?)\s*\)\s*;", source, re.DOTALL)

        if port_section:
            text = port_section.group(1)
            for line in text.split(","):
                line = line.strip()
                if not line:
                    continue
                port_match = re.match(
                    r"(input|output|inout)\s+"
                    r"(?:(logic|wire|reg)\s+)?"
                    r"(signed\s+)?"
                    r"(?:\[(\d+):(\d+)\]\s+)?"
                    r"(\w+)"
                    r"(?:\s+\[0:(\d+)\])?",
                    line,
                )
                if port_match:
                    direction = PortDirection(port_match.group(1))
                    port_type = (
                        PortType(port_match.group(2)) if port_match.group(2) else PortType.LOGIC
                    )
                    is_signed = port_match.group(3) is not None
                    if port_match.group(4) and port_match.group(5):
                        width = int(port_match.group(4)) - int(port_match.group(5)) + 1
                    else:
                        width = 1
                    port_name = port_match.group(6)
                    is_array = port_match.group(7) is not None
                    array_size = int(port_match.group(7)) + 1 if is_array else 0
                    ports.append(
                        ModulePort(
                            port_name,
                            direction,
                            port_type,
                            width,
                            is_signed,
                            is_array,
                            array_size,
                        )
                    )

        return cls(name=name, ports=ports, params=params)

    @property
    def input_ports(self) -> List[ModulePort]:
        """Input ports excluding generated clock and reset controls."""
        return [
            port
            for port in self.ports
            if port.direction == PortDirection.INPUT and not port.is_clock and not port.is_reset
        ]

    @property
    def output_ports(self) -> List[ModulePort]:
        """Output ports monitored by the generated scoreboard and coverage."""
        return [port for port in self.ports if port.direction == PortDirection.OUTPUT]

    @property
    def clock_port(self) -> Optional[ModulePort]:
        """First parsed clock-like port, if the RTL module declares one."""
        return next((port for port in self.ports if port.is_clock), None)

    @property
    def reset_port(self) -> Optional[ModulePort]:
        """First parsed reset-like port, if the RTL module declares one."""
        return next((port for port in self.ports if port.is_reset), None)

    @property
    def total_input_bits(self) -> int:
        """Total data-input width excluding clock and reset ports."""
        return sum(port.width for port in self.input_ports)

    @property
    def total_output_bits(self) -> int:
        """Total output width observed by generated verification components."""
        return sum(port.width for port in self.output_ports)
