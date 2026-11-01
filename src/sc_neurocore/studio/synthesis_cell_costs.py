# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FPGA primitive resource costs

"""Count registered FPGA primitives while preserving unknown hardware costs."""

from __future__ import annotations

_NO_COST: dict[str, int] = {}

#: What each primitive a family's Yosys flow emits takes of the judged
#: resources. I/O buffers, constants, clock buffers, carry cells that ride in
#: a LUT's logic cell and wide-function multiplexers take none. A cell this
#: table does not know (a family's distributed RAM, a hand-placed primitive)
#: is reported and leaves the design without a fit verdict unless the counted
#: cells already overflow: its cost is unknown, so the counts are a floor.
#: The portless $scopeinfo debug metadata emitted on hierarchy flattening takes
#: no device resources. Names were checked against Yosys 0.33/0.70 output; the substring
#: matching it replaces counted Gowin wide multiplexers as LUTs, Gowin and
#: Xilinx LUT-RAM as block RAM, and missed ECP5 carry LUTs and DP16KD.
_CELL_COST: dict[str, dict[str, dict[str, int]]] = {
    "ice40": {
        "$scopeinfo": _NO_COST,
        "SB_LUT4": {"luts": 1},
        "SB_CARRY": _NO_COST,
        "SB_RAM40_4K": {"brams": 1},
        "SB_RAM40_4KNR": {"brams": 1},
        "SB_RAM40_4KNW": {"brams": 1},
        "SB_RAM40_4KNRNW": {"brams": 1},
        "SB_MAC16": {"dsps": 1},
        "SB_IO": _NO_COST,
        "SB_GB": _NO_COST,
        "SB_GB_IO": _NO_COST,
    },
    "ecp5": {
        "$scopeinfo": _NO_COST,
        "LUT4": {"luts": 1},
        "CCU2C": {"luts": 2},
        "PFUMX": _NO_COST,
        "L6MUX21": _NO_COST,
        "TRELLIS_FF": {"ffs": 1},
        "DP16KD": {"brams": 1},
        "PDPW16KD": {"brams": 1},
        "MULT18X18D": {"dsps": 1},
        "TRELLIS_IO": _NO_COST,
        "DCCA": _NO_COST,
        "VHI": _NO_COST,
        "VLO": _NO_COST,
    },
    "gowin": {
        "$scopeinfo": _NO_COST,
        **{f"LUT{width}": {"luts": 1} for width in range(1, 5)},
        "ALU": {"luts": 1},
        **{f"MUX2_LUT{width}": _NO_COST for width in range(5, 9)},
        **{name: {"brams": 1} for name in ("SP", "SPX9", "SDP", "SDPX9", "SDPB", "SDPX9B")},
        **{name: {"brams": 1} for name in ("DP", "DPX9", "DPB", "DPX9B", "ROM", "ROMX9")},
        **{name: {"brams": 1} for name in ("pROM", "pROMX9")},
        "MULT18X18": {"dsps": 1},
        "MULT36X36": {"dsps": 4},
        **{name: _NO_COST for name in ("IBUF", "OBUF", "IOBUF", "TBUF", "GND", "VCC")},
    },
    "xilinx": {
        "$scopeinfo": _NO_COST,
        **{f"LUT{width}": {"luts": 1} for width in range(1, 7)},
        "LUT6_2": {"luts": 1},
        "INV": {"luts": 1},
        "SRL16E": {"luts": 1},
        "SRLC32E": {"luts": 1},
        # Distributed RAM, in LUTs per primitive (Xilinx UG474).
        "RAM32X1S": {"luts": 1},
        "RAM64X1S": {"luts": 1},
        "RAM128X1S": {"luts": 2},
        "RAM256X1S": {"luts": 4},
        "RAM32X1D": {"luts": 2},
        "RAM64X1D": {"luts": 2},
        "RAM128X1D": {"luts": 4},
        "RAM32M": {"luts": 4},
        "RAM64M": {"luts": 4},
        "CARRY4": _NO_COST,
        "MUXF7": _NO_COST,
        "MUXF8": _NO_COST,
        **{name: {"ffs": 1} for name in ("FDRE", "FDSE", "FDCE", "FDPE", "LDCE", "LDPE")},
        "RAMB18E1": {"brams": 1},
        "RAMB36E1": {"brams": 2},
        "DSP48E1": {"dsps": 1},
        **{name: _NO_COST for name in ("IBUF", "OBUF", "OBUFT", "IOBUF", "BUFG", "BUFGCTRL")},
        **{name: _NO_COST for name in ("GND", "VCC")},
    },
}


def _flip_flop_cost(target: str, cell_type: str) -> dict[str, int] | None:
    """Recognise a family's flip-flop variants, which the tables do not list."""
    if target == "ice40" and cell_type.startswith("SB_DFF"):
        return {"ffs": 1}
    if target == "gowin" and cell_type.startswith("DFF"):
        return {"ffs": 1}
    return None


def cell_cost(target: str, cell_type: str) -> dict[str, int] | None:
    """Return what one cell takes of the judged resources, or None if unknown.

    Parameters
    ----------
    target : str
        Target identifier.
    cell_type : str
        The Yosys cell type.

    Returns
    -------
    dict[str, int] or None
        Resource amounts, empty for a cell that takes none, None for a cell
        whose cost the Studio does not know.
    """
    known = _CELL_COST.get(target, {}).get(cell_type)
    return known if known is not None else _flip_flop_cost(target, cell_type)
