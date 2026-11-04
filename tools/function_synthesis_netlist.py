# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Typed combinational synthesis netlists

"""Read the actual words, ports and primitive connections used by a proof."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from tools.function_synthesis_json import decode_native_json

Bit: TypeAlias = int | str
Direction: TypeAlias = Literal["input", "output"]
_MAX_NETLIST_BYTES = 64 * 1024 * 1024
_MAX_VECTOR_BITS = 1024 * 1024


def object_mapping(value: object) -> dict[str, object]:
    """Require a JSON object with string keys before inspecting its fields."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("A synthesis netlist field must be a JSON object")
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _text_mapping(value: object) -> dict[str, str]:
    """Keep exact string parameters and refuse implicit numeric conversions."""
    mapping = object_mapping(value)
    if not all(isinstance(item, str) for item in mapping.values()):
        raise ValueError("Synthesis parameters and directions must contain strings")
    return {key: item for key, item in mapping.items() if isinstance(item, str)}


def _bits(value: object) -> tuple[Bit, ...]:
    """Validate actual wire identifiers and four-state constant literals."""
    if not isinstance(value, list) or not value or len(value) > _MAX_VECTOR_BITS:
        raise ValueError("A synthesis word must have a bounded positive width")
    result: list[Bit] = []
    for bit in value:
        if isinstance(bit, bool):
            raise ValueError("A synthesis wire identifier must not be a boolean")
        if (isinstance(bit, int) and bit >= 2) or (
            isinstance(bit, str) and bit in {"0", "1", "x", "z"}
        ):
            result.append(bit)
        else:
            raise ValueError("A synthesis word contains an invalid wire or constant")
    return tuple(result)


def _direction(value: object) -> Direction:
    """Admit only unidirectional combinational connections."""
    if value == "input":
        return "input"
    if value == "output":
        return "output"
    raise ValueError("A function proof requires input or output port directions")


def parameter_integer(parameters: Mapping[str, str], name: str) -> int:
    """Read one exact binary Yosys parameter without signed decimal coercion.

    Parameters
    ----------
    parameters : mapping of str to str
        Actual parameter strings from the selected cell.
    name : str
        Required binary-valued parameter.

    Returns
    -------
    int
        Nonnegative value of the complete binary representation.

    Raises
    ------
    ValueError
        If the parameter is absent, empty or not a binary bit string.

    """
    value = parameters.get(name)
    if value is None or not value or set(value) - {"0", "1"}:
        raise ValueError("A required synthesis parameter is not an exact binary integer")
    return int(value, 2)


@dataclass(frozen=True, slots=True)
class NetlistPort:
    """Retain the direction and actual ordered bits of a module port.

    Parameters
    ----------
    direction : {'input', 'output'}
        Actual direction declared by Yosys.
    bits : tuple of int or str
        Wire identifiers or exact four-state constant literals, least bit first.

    """

    direction: Direction
    bits: tuple[Bit, ...]


@dataclass(frozen=True, slots=True)
class NetlistCell:
    """Retain exact cell semantics and all ordered port connections.

    Parameters
    ----------
    kind : str
        Actual Yosys cell or mapped primitive type.
    parameters : dict of str to str
        Complete unmodified parameter representation.
    directions : dict of str to {'input', 'output'}
        Actual direction of every connected port.
    connections : dict of str to tuple of int or str
        Actual least-bit-first connection vectors; no port may be omitted.

    """

    kind: str
    parameters: dict[str, str]
    directions: dict[str, Direction]
    connections: dict[str, tuple[Bit, ...]]

    @classmethod
    def read(cls, value: object) -> NetlistCell:
        """Validate a cell's complete parameter and connected-port records."""
        data = object_mapping(value)
        kind = data.get("type")
        if not isinstance(kind, str) or not kind:
            raise ValueError("A synthesis cell requires its actual nonempty type")
        parameters = _text_mapping(data.get("parameters"))
        directions = {
            name: _direction(direction)
            for name, direction in object_mapping(data.get("port_directions")).items()
        }
        connections = {
            name: _bits(bits) for name, bits in object_mapping(data.get("connections")).items()
        }
        if not directions or directions.keys() != connections.keys():
            raise ValueError("Every connected synthesis port requires an exact direction")
        return cls(kind, parameters, directions, connections)


@dataclass(frozen=True, slots=True)
class Netlist:
    """Bind one actual function module to its typed graph and original JSON.

    Parameters
    ----------
    ports : dict of str to NetlistPort
        Complete original primary interface.
    cells : dict of str to NetlistCell
        Actual mapped or word-level functions.
    names : dict of str to tuple of int or str
        Actual named wire vectors, used only as checked correspondence anchors.
    offsets : dict of str to int
        Original vector offsets, separately checked when matching named vectors.
    memory_identity : str
        Canonical JSON representation of retained memory declarations.
    raw : dict of str to object
        Original module record for exporting proof clones without rewriting cells.

    """

    ports: dict[str, NetlistPort]
    cells: dict[str, NetlistCell]
    names: dict[str, tuple[Bit, ...]]
    offsets: dict[str, int]
    memory_identity: str
    raw: dict[str, object]

    @classmethod
    def read(cls, path: Path) -> Netlist:
        """Read a bounded actual function module and validate its graph fields.

        Parameters
        ----------
        path : pathlib.Path
            Regular, non-symlink Yosys JSON artifact containing function_unit.

        Returns
        -------
        Netlist
            Exact primary ports, parameters, directions and ordered bit vectors.

        Raises
        ------
        OSError, ValueError
            For unavailable, linked, oversized, malformed or incomplete records.

        """
        if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_NETLIST_BYTES:
            raise ValueError("A function netlist must be a bounded regular artifact")
        with path.open("rb") as stream:
            raw = stream.read(_MAX_NETLIST_BYTES + 1)
        if len(raw) > _MAX_NETLIST_BYTES:
            raise ValueError("A function netlist must be a bounded regular artifact")
        value = decode_native_json(raw)
        modules = object_mapping(object_mapping(value).get("modules"))
        if "function_unit" not in modules:
            raise ValueError("A synthesis artifact must contain the actual function module")
        data = object_mapping(modules["function_unit"])
        ports: dict[str, NetlistPort] = {}
        for name, record in object_mapping(data.get("ports")).items():
            port = object_mapping(record)
            ports[name] = NetlistPort(_direction(port.get("direction")), _bits(port.get("bits")))
        cells = {
            name: NetlistCell.read(cell) for name, cell in object_mapping(data.get("cells")).items()
        }
        names: dict[str, tuple[Bit, ...]] = {}
        offsets: dict[str, int] = {}
        for name, record in object_mapping(data.get("netnames")).items():
            net = object_mapping(record)
            offset = net.get("offset", 0)
            if isinstance(offset, bool) or not isinstance(offset, int):
                raise ValueError("A synthesis vector offset must be an integer")
            names[name], offsets[name] = _bits(net.get("bits")), offset
        if any(
            name in names and (names[name] != port.bits or offsets[name] != 0)
            for name, port in ports.items()
        ):
            raise ValueError("A named original port differs from its declared wire connections")
        memories = object_mapping(data.get("memories", {}))
        return cls(
            ports,
            cells,
            names,
            offsets,
            json.dumps(memories, sort_keys=True, separators=(",", ":")),
            data,
        )

    def primary_inputs(self, widths: Mapping[str, int], output_width: int) -> dict[int, str]:
        """Require every original input bit to remain independently driven.

        Parameters
        ----------
        widths : mapping of str to int
            All original argument names and their complete declared widths.
        output_width : int
            Complete original result width, including saturated upper bits.

        Returns
        -------
        dict of int to str
            Independent actual input wires and their canonical original names.

        Raises
        ------
        ValueError
            For changed interfaces, constant or aliased original inputs.

        """
        expected = {name: ("input", width) for name, width in widths.items()}
        expected["result"] = ("output", output_width)
        if not widths or any(
            isinstance(width, bool) or not 1 <= width <= 64 for width in widths.values()
        ):
            raise ValueError("Original argument widths must be positive supported words")
        if (
            isinstance(output_width, bool)
            or not 1 <= output_width <= 64
            or {name: (port.direction, len(port.bits)) for name, port in self.ports.items()}
            != expected
        ):
            raise ValueError("The complete original function interface changed")
        result: dict[int, str] = {}
        for name in sorted(widths):
            for index, bit in enumerate(self.ports[name].bits):
                if not isinstance(bit, int) or bit in result:
                    raise ValueError("Original primary input bits were constrained or aliased")
                result[bit] = f"{name}[{index}]"
        return result

    def clone_module(self) -> dict[str, object]:
        """Return an independent exact JSON module for an explicitly described proof clone."""
        return copy.deepcopy(self.raw)
