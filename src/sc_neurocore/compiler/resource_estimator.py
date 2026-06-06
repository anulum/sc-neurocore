# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Resource estimation

"""Resource estimation utilities for compiled neuron modules.

Estimates LUT/FF/DSP/BRAM usage from Verilog source without synthesis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ResourceEstimate:
    """Estimated FPGA resource usage.

    Attributes
    ----------
    luts : int
        Estimated look-up tables.
    ffs : int
        Estimated flip-flops (registers).
    dsps : int
        Estimated DSP blocks.
    brams : int
        Estimated block RAMs.
    mul_count : int
        Number of multiplications in the design.
    add_count : int
        Number of additions/subtractions.
    reg_bits : int
        Total register bits.
    """

    luts: int
    ffs: int
    dsps: int
    brams: int
    mul_count: int
    add_count: int
    reg_bits: int


def estimate_resources(
    verilog: str,
    *,
    data_width: int = 16,
    has_dsp: bool = True,
) -> ResourceEstimate:
    """Estimate FPGA resources from generated Verilog without synthesis.

    Uses pattern matching on the Verilog source to count multipliers, adders,
    registers, and LUTs. This is a heuristic — actual usage depends on the
    synthesis tool, but estimates are within ~20% for typical designs.

    Parameters
    ----------
    verilog : str
        Generated Verilog source code.
    data_width : int
        Neuron data width (for LUT estimation).
    has_dsp : bool
        True if target has DSP blocks (multiplies go to DSP, not LUTs).

    Returns
    -------
    ResourceEstimate
        Estimated resource usage.
    """
    mul_count = len(re.findall(r"wire\s+signed\s+\[.*?\]\s+_mul\d+", verilog))
    add_count = verilog.count(" + ") + verilog.count(" - ")
    reg_count = len(re.findall(r"reg\s+signed\s+\[", verilog))
    reg_bits = reg_count * data_width

    # LUT estimation heuristics
    luts_per_add = data_width  # 1 LUT per bit for addition
    luts_per_mul = 0 if has_dsp else (data_width * data_width // 4)
    luts_per_mux = data_width // 2  # For saturation/threshold muxes
    mux_count = verilog.count("?")  # Ternary operators

    luts = add_count * luts_per_add + mul_count * luts_per_mul + mux_count * luts_per_mux

    ffs = reg_bits + data_width  # + spike_out + control

    dsps = mul_count if has_dsp else 0

    # BRAM: 0 for single neuron (registers only)
    brams = 0

    return ResourceEstimate(
        luts=max(luts, 1),
        ffs=max(ffs, 1),
        dsps=dsps,
        brams=brams,
        mul_count=mul_count,
        add_count=add_count,
        reg_bits=reg_bits,
    )
