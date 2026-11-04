# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete native DSP port configuration and source alignment

"""Qualify complete DSP pins and retain checked source-clone interface extensions."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import TypeAlias

from tools.function_synthesis_json import MAX_NETLIST_BYTES
from tools.function_synthesis_netlist import Netlist, NetlistCell, object_mapping, parameter_integer

MacroConfiguration: TypeAlias = tuple[str, str, str]
_DSP_REGISTERS = frozenset(
    {
        "ACASCREG",
        "ADREG",
        "ALUMODEREG",
        "AREG",
        "BCASCREG",
        "BREG",
        "CARRYINREG",
        "CARRYINSELREG",
        "CREG",
        "DREG",
        "INMODEREG",
        "MREG",
        "OPMODEREG",
        "PREG",
    }
)
_DSP_MODES: frozenset[MacroConfiguration] = frozenset(
    {
        ("0000101", "DIRECT", "DIRECT"),
        ("0010101", "DIRECT", "DIRECT"),
        ("1010101", "CASCADE", "DIRECT"),
        ("0110101", "DIRECT", "DIRECT"),
        ("1010101", "DIRECT", "CASCADE"),
    }
)
_DSP_CORE_WIDTHS = {
    "A": 30,
    "B": 18,
    "ACIN": 30,
    "BCIN": 18,
    "C": 48,
    "PCIN": 48,
    "D": 25,
    "CARRYIN": 1,
    "INMODE": 5,
    "ALUMODE": 4,
    "OPMODE": 7,
    "CARRYINSEL": 3,
    "P": 48,
    "ACOUT": 30,
    "BCOUT": 18,
    "PCOUT": 48,
}
DSP_AUXILIARY_INPUTS = (
    "CARRYCASCIN",
    "CEA1",
    "CEA2",
    "CEAD",
    "CEALUMODE",
    "CEB1",
    "CEB2",
    "CEC",
    "CECARRYIN",
    "CECTRL",
    "CED",
    "CEINMODE",
    "CEM",
    "CEP",
    "CLK",
    "MULTSIGNIN",
    "RSTA",
    "RSTALLCARRYIN",
    "RSTALUMODE",
    "RSTB",
    "RSTC",
    "RSTCTRL",
    "RSTD",
    "RSTINMODE",
    "RSTM",
    "RSTP",
)
_DSP_STATUS_WIDTHS = {
    "CARRYCASCOUT": 1,
    "CARRYOUT": 4,
    "MULTSIGNOUT": 1,
    "OVERFLOW": 1,
    "PATTERNBDETECT": 1,
    "PATTERNDETECT": 1,
    "UNDERFLOW": 1,
}
_DSP_WIDTHS = {
    **_DSP_CORE_WIDTHS,
    **dict.fromkeys(DSP_AUXILIARY_INPUTS, 1),
    **_DSP_STATUS_WIDTHS,
}
_DSP_OUTPUTS = frozenset({"P", "PCOUT", "ACOUT", "BCOUT", *_DSP_STATUS_WIDTHS})


def configured_dsp(cell: NetlistCell) -> MacroConfiguration:
    """Require one fully specified, natively qualified combinational DSP mode.

    Parameters
    ----------
    cell : NetlistCell
        Actual connected DSP48E1 primitive, including every parameter and port.

    Returns
    -------
    tuple of str
        OPMODE, A_INPUT and B_INPUT for the exact admitted configuration.

    Raises
    ------
    ValueError
        For state, unproved controls, unknown values, ports or configurations.

    """
    parameters, connections = cell.parameters, cell.connections
    allowed = _DSP_REGISTERS | {"A_INPUT", "B_INPUT", "USE_DPORT", "USE_MULT", "USE_SIMD"}
    if (
        cell.kind != "DSP48E1"
        or parameters.keys() != allowed
        or any(parameter_integer(parameters, name) for name in _DSP_REGISTERS)
    ):
        raise ValueError("A DSP proof requires the complete qualified zero-register configuration")
    if (parameters["USE_DPORT"], parameters["USE_MULT"], parameters["USE_SIMD"]) != (
        "FALSE",
        "MULTIPLY",
        "ONE48",
    ):
        raise ValueError("The connected DSP arithmetic configuration is not qualified")
    required = _DSP_CORE_WIDTHS.keys() - {"ACOUT", "BCOUT", "PCOUT"}
    auxiliary = connections.keys() - _DSP_CORE_WIDTHS.keys()
    if (
        not required <= connections.keys()
        or (auxiliary and not set(DSP_AUXILIARY_INPUTS) <= connections.keys())
        or any(
            port not in _DSP_WIDTHS or len(bits) != _DSP_WIDTHS[port]
            for port, bits in connections.items()
        )
    ):
        raise ValueError("The connected DSP port geometry is not qualified")
    if any(
        direction != ("output" if port in _DSP_OUTPUTS else "input")
        for port, direction in cell.directions.items()
    ):
        raise ValueError("A configured DSP port has a different direction")
    if any(
        isinstance(bit, str) and bit not in {"0", "1"}
        for port, bits in connections.items()
        if cell.directions[port] == "input"
        for bit in bits
    ):
        raise ValueError("A configured DSP input contains an unknown value")
    if any(
        set(connections[port]) != {"0"}
        for port in ["INMODE", "ALUMODE", "CARRYINSEL", "CARRYIN", "D"]
    ):
        raise ValueError("The connected DSP control is not qualified")
    if any(set(connections[port]) != {"0"} for port in DSP_AUXILIARY_INPUTS if port in connections):
        raise ValueError("DSP auxiliary controls must be known zero in this native profile")
    configuration = (
        "".join(reversed(tuple(str(bit) for bit in connections["OPMODE"]))),
        parameters["A_INPUT"],
        parameters["B_INPUT"],
    )
    if configuration not in _DSP_MODES:
        raise ValueError("The configured DSP mode has no qualified arithmetic contract")
    return configuration


def _identity(module: Netlist) -> str:
    """Bind every original module field independently of JSON presentation."""
    text = json.dumps(module.raw, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def align_dsp_ports(word: Netlist, physical: Netlist) -> tuple[Netlist, dict[str, object]]:
    """Extend only derived source pins whose actual native addition is qualified.

    Parameters
    ----------
    word, physical : Netlist
        Complete original post-DSP and physical modules. Neither is modified.
        The caller separately validates their complete original primary interface.

    Returns
    -------
    tuple of Netlist and dict of str to object
        Derived source module and deterministic original/derived identity record.
        Added input constants require the complete universal-control arithmetic
        contract; added output values are fresh, universally free cut values.

    Raises
    ------
    ValueError
        For unknown configurations, changed existing pins or any added output
        observed by a primary port, another driver or any actual cell input.

    """
    observed = {bit for port in physical.ports.values() for bit in port.bits}
    observed.update(
        bit
        for cell in physical.cells.values()
        for port, direction in cell.directions.items()
        if direction == "input"
        for bit in cell.connections[port]
    )
    output_counts = Counter(
        bit
        for cell in physical.cells.values()
        for port, direction in cell.directions.items()
        if direction == "output"
        for bit in cell.connections[port]
    )
    original_bits = [
        bit
        for cell in word.cells.values()
        for bits in cell.connections.values()
        for bit in bits
        if isinstance(bit, int)
    ]
    original_bits.extend(
        bit for bits in word.names.values() for bit in bits if isinstance(bit, int)
    )
    original_bits.extend(
        bit for port in word.ports.values() for bit in port.bits if isinstance(bit, int)
    )
    fresh = max(original_bits, default=1) + 1
    clone = word.clone_module()
    cells = object_mapping(clone["cells"])
    additions: dict[str, object] = {}
    for name, target in sorted(physical.cells.items()):
        if target.kind != "DSP48E1":
            continue
        original = word.cells.get(name)
        if original is None or original.kind != "DSP48E1":
            raise ValueError("The physical DSP instances do not match the actual word checkpoint")
        configured_dsp(original)
        configured_dsp(target)
        if (
            original.parameters != target.parameters
            or not original.connections.keys() <= target.connections.keys()
        ):
            raise ValueError("A native DSP alignment changed an existing parameter or port")
        ports: dict[str, object] = {}
        raw = object_mapping(cells[name])
        raw_connections = object_mapping(raw["connections"])
        raw_directions = object_mapping(raw["port_directions"])
        for port in sorted(target.connections.keys() - original.connections.keys()):
            bits = target.connections[port]
            direction = target.directions[port]
            if direction == "input":
                source_bits = list(bits)
            else:
                if any(
                    not isinstance(bit, int) or bit in observed or output_counts[bit] != 1
                    for bit in bits
                ):
                    raise ValueError("An added native DSP output is observed, fixed or aliased")
                source_bits = list(range(fresh, fresh + len(bits)))
                fresh += len(bits)
            raw_connections[port] = source_bits
            raw_directions[port] = direction
            ports[port] = {
                "direction": direction,
                "source_bits": source_bits,
                "physical_bits": list(bits),
            }
        raw["connections"], raw["port_directions"] = raw_connections, raw_directions
        cells[name] = raw
        if ports:
            additions[name] = ports
    clone["cells"] = cells
    aligned = replace(
        word,
        ports=word.ports.copy(),
        cells={name: NetlistCell.read(cell) for name, cell in cells.items()},
        names=word.names.copy(),
        offsets=word.offsets.copy(),
        raw=clone,
    )
    return aligned, {
        "schema_version": "sc-neurocore.native-dsp-port-alignment.v1",
        "source_module_sha256": _identity(word),
        "physical_module_sha256": _identity(physical),
        "aligned_source_module_sha256": _identity(aligned),
        "added_ports": additions,
    }


def read_dsp_port_alignment(word: Netlist, physical: Netlist, directory: Path) -> Netlist:
    """Recompute a retained interface derivation from unchanged original modules.

    Parameters
    ----------
    word, physical : Netlist
        Actual complete original source and physical modules.
    directory : pathlib.Path
        Native residual proof directory containing the derived record and clone.

    Returns
    -------
    Netlist
        Recomputed complete source clone after exact record and graph equality.

    Raises
    ------
    OSError, ValueError
        For missing, linked, oversized, corrupt or changed alignment artifacts.

    """
    record = directory / "native-port-alignment.json"
    source = directory / "aligned-word.json"
    if directory.is_symlink() or any(
        path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_NETLIST_BYTES
        for path in (record, source)
    ):
        raise ValueError("Retained DSP port alignment requires bounded regular artifacts")
    aligned, expected = align_dsp_ports(word, physical)
    if json.loads(record.read_bytes()) != expected or Netlist.read(source).raw != aligned.raw:
        raise ValueError("Retained DSP port alignment differs from its actual original modules")
    return aligned
