# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Memory map generator

"""Address decoder generation for multi-neuron SoC arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MemoryMap:
    """Address decoder specification for neuron arrays.

    Attributes
    ----------
    base_address : int
        Base address of neuron array.
    entries : list[dict[str, int | str]]
        Address map entries.
    total_bytes : int
        Total address space consumed.
    decoder_verilog : str
        Generated address decoder Verilog.
    """

    base_address: int
    entries: list[dict[str, Any]]
    total_bytes: int
    decoder_verilog: str


def generate_memory_map(
    module_name: str,
    equations: dict[str, str],
    *,
    num_neurons: int = 256,
    data_width: int = 16,
    base_address: int = 0x1000_0000,
) -> MemoryMap:
    """Generate address decoder for multi-neuron SoC arrays.

    Parameters
    ----------
    module_name : str
        Module name.
    equations : dict[str, str]
        ODE equations.
    num_neurons : int
        Number of neuron instances.
    data_width : int
        Register width in bits.
    base_address : int
        Base address.

    Returns
    -------
    MemoryMap
    """
    vars_list = list(equations.keys())
    bytes_per_reg = max(2, data_width // 8)
    regs_per_neuron = len(vars_list) + 1
    stride = regs_per_neuron * bytes_per_reg

    entries = []
    for n in range(min(num_neurons, 8)):
        for i, sv in enumerate(vars_list):
            addr = base_address + n * stride + i * bytes_per_reg
            entries.append(
                {
                    "address": addr,
                    "name": f"neuron_{n}_{sv}",
                    "width": data_width,
                }
            )
        ctrl_addr = base_address + n * stride + len(vars_list) * bytes_per_reg
        entries.append(
            {
                "address": ctrl_addr,
                "name": f"neuron_{n}_ctrl",
                "width": data_width,
            }
        )

    total = num_neurons * stride
    verilog = [
        f"// Address decoder for {module_name} — {num_neurons} neurons",
        f"// Base: 0x{base_address:08X}, Stride: {stride} bytes",
        f"module {module_name}_addr_dec (",
        f"    input  [{data_width - 1}:0] addr,",
        f"    output reg [{len(vars_list)}:0] reg_sel,",
        f"    output reg [{num_neurons.bit_length() - 1}:0] neuron_sel",
        ");",
        f"    wire [{num_neurons.bit_length() - 1}:0] idx = "
        f"(addr - 32'h{base_address:08X}) / {stride};",
        f"    wire [{regs_per_neuron.bit_length() - 1}:0] reg_off = "
        f"((addr - 32'h{base_address:08X}) % {stride}) / {bytes_per_reg};",
        "    always @(*) begin",
        "        neuron_sel = idx;",
        "        reg_sel = reg_off;",
        "    end",
        "endmodule",
    ]

    return MemoryMap(
        base_address=base_address,
        entries=entries,
        total_bytes=total,
        decoder_verilog="\n".join(verilog),
    )
