# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Historical bus-interface compatibility facade

"""Generate SoC bus wrappers and live-control parameter-bank RTL.

The historical :mod:`sc_neurocore.hdl_gen.bus_interface` module remains the
public boundary. Protocol-specific renderers live in bounded private modules so
legacy imports, signatures, qualified names, and generated source remain
stable.
"""

from __future__ import annotations

from typing import Literal

from sc_neurocore.compiler.live_control import MMIOUpdateSpec

from ._bus_wrappers import (
    render_axi_lite_wrapper as _render_axi_lite_wrapper,
    render_wishbone_wrapper as _render_wishbone_wrapper,
)
from ._live_parameter_bank import (
    render_axi_live_parameter_bank as _render_axi_live_parameter_bank,
)
from ._pcie_live_parameter_bank import (
    render_pcie_live_parameter_bank as _render_pcie_live_parameter_bank,
)


BusProtocol = Literal["axi_lite", "wishbone"]

__all__ = [
    "BusProtocol",
    "generate_bus_wrapper",
    "generate_live_parameter_bank",
    "generate_register_map",
]


def generate_bus_wrapper(
    inner_module: str,
    params: dict[str, int],
    *,
    bus: BusProtocol = "axi_lite",
    data_width: int = 16,
    addr_width: int = 8,
    bus_data_width: int = 32,
    base_address: int = 0,
) -> str:
    """Generate a bus-attached wrapper around a compiled neuron module.

    Parameters
    ----------
    inner_module : str
        Name of the inner Verilog neuron module (for example, ``"sc_lif"``).
    params : dict[str, int]
        Mapping from Verilog parameter name to bit width.
    bus : BusProtocol
        Bus protocol: ``"axi_lite"`` or ``"wishbone"``.
    data_width : int
        Neuron fixed-point data width.
    addr_width : int
        Address bus width.
    bus_data_width : int
        Bus register data width.
    base_address : int
        Reserved documentation address retained for API compatibility.

    Returns
    -------
    str
        Complete SystemVerilog source for the bus wrapper module.
    """
    if bus == "axi_lite":
        return _render_axi_lite_wrapper(
            inner_module,
            params,
            data_width,
            addr_width,
            bus_data_width,
        )
    if bus == "wishbone":
        return _render_wishbone_wrapper(
            inner_module,
            params,
            data_width,
            addr_width,
            bus_data_width,
        )
    raise ValueError(f"Unsupported bus protocol: {bus!r}. Use 'axi_lite' or 'wishbone'.")


def generate_register_map(
    params: dict[str, int],
    *,
    base_address: int = 0,
) -> dict[str, int]:
    """Return the register map for a neuron's parameters.

    Parameters
    ----------
    params : dict[str, int]
        Parameter names and their bit widths.
    base_address : int
        Starting byte address.

    Returns
    -------
    dict[str, int]
        Mapping from register name to byte address.
    """
    reg_map: dict[str, int] = {}
    offset = base_address
    reg_map["CTRL"] = offset
    offset += 4
    reg_map["I_T"] = offset
    offset += 4
    reg_map["SPIKE_COUNT"] = offset
    offset += 4
    for name in params:
        reg_map[name] = offset
        offset += 4
    return reg_map


def generate_live_parameter_bank(
    spec: MMIOUpdateSpec,
    *,
    module_name: str = "sc_live_parameter_bank",
    addr_width: int | None = None,
    bus_data_width: int = 32,
    block_ram_threshold_bits: int = 1024,
) -> str:
    """Generate a live-parameter bank from an MMIO update spec.

    The emitted RTL stores each parameter bank in distributed RAM or BRAM and
    exposes the fixed live-control register map through either AXI4-Lite or a
    PCIe MMIO register-window adapter. The PCIe path models the endpoint
    adapter contract: upstream PCIe hard IP decodes posted writes and reads
    into the single-clock MMIO strobes exposed here.

    Parameters
    ----------
    spec : MMIOUpdateSpec
        Validated live-control register and parameter-bank contract.
    module_name : str
        SystemVerilog module identifier.
    addr_width : int or None
        Optional address-width override.
    bus_data_width : int
        Bus data width. Maintained live-control paths require 32 bits.
    block_ram_threshold_bits : int
        Minimum bank capacity that receives a block-RAM style hint.

    Returns
    -------
    str
        Complete SystemVerilog source for the selected live-control protocol.
    """
    if spec.bus_protocol == "pcie":
        return _render_pcie_live_parameter_bank(
            spec,
            module_name=module_name,
            addr_width=addr_width,
            bus_data_width=bus_data_width,
            block_ram_threshold_bits=block_ram_threshold_bits,
        )
    return _render_axi_live_parameter_bank(
        spec,
        module_name=module_name,
        addr_width=addr_width,
        bus_data_width=bus_data_width,
        block_ram_threshold_bits=block_ram_threshold_bits,
    )
