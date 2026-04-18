# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FPGA target models for energy estimation

"""FPGA resource and power models calibrated against synthesis reports.

LUT counts calibrated against Yosys synth_ice40 reports for SC-NeuroCore
HDL modules. Power models use C_eff × V² × f × activity estimation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FPGATarget:
    """FPGA target specification."""

    name: str
    family: str
    total_luts: int
    total_bram_kb: int
    total_dsp: int
    voltage: float  # core voltage (V)
    max_freq_mhz: float  # typical max frequency
    c_eff_per_lut_ff: float  # effective capacitance per LUT toggle (fF)


# Calibrated against Yosys synthesis reports
ICE40_HX8K = FPGATarget(
    name="ice40_hx8k",
    family="ice40",
    total_luts=7680,
    total_bram_kb=128,
    total_dsp=0,
    voltage=1.2,
    max_freq_mhz=150.0,
    c_eff_per_lut_ff=5.0,
)

ECP5_85K = FPGATarget(
    name="ecp5_85k",
    family="ecp5",
    total_luts=83640,
    total_bram_kb=3744,
    total_dsp=156,
    voltage=1.1,
    max_freq_mhz=400.0,
    c_eff_per_lut_ff=4.0,
)

ARTIX7_100T = FPGATarget(
    name="artix7_100t",
    family="xc7a",
    total_luts=63400,
    total_bram_kb=4860,
    total_dsp=240,
    voltage=1.0,
    max_freq_mhz=450.0,
    c_eff_per_lut_ff=3.5,
)

ZYNQ_7020 = FPGATarget(
    name="zynq_7020",
    family="xc7z",
    total_luts=53200,
    total_bram_kb=4480,
    total_dsp=220,
    voltage=1.0,
    max_freq_mhz=400.0,
    c_eff_per_lut_ff=3.5,
)

TARGETS = {
    "ice40": ICE40_HX8K,
    "ecp5": ECP5_85K,
    "artix7": ARTIX7_100T,
    "zynq": ZYNQ_7020,
}


# SC module resource costs (calibrated from Yosys synth_ice40 reports)
# Format: (luts, ffs, bram_bits)
@dataclass(frozen=True)
class ModuleCost:
    """Resource cost for one SC module instance."""

    luts: int
    ffs: int
    bram_bits: int = 0
    description: str = ""


# Per-neuron costs (from Yosys synthesis of sc_lif_neuron.v)
LIF_NEURON = ModuleCost(luts=120, ffs=48, description="Q8.8 LIF neuron")
EVENT_NEURON = ModuleCost(luts=85, ffs=32, description="Event-driven LIF")

# Per-encoder costs
BITSTREAM_ENCODER = ModuleCost(luts=40, ffs=16, description="LFSR encoder")

# Per-synapse costs (SC: 1 AND gate = 1 LUT)
SC_SYNAPSE = ModuleCost(luts=1, ffs=0, description="SC AND synapse")

# Infrastructure
AXI_LITE = ModuleCost(luts=200, ffs=64, description="AXI-Lite register file")
AXI_STREAM = ModuleCost(luts=150, ffs=48, description="AXI-Stream interface")
DMA = ModuleCost(luts=300, ffs=96, description="DMA controller")
AER_ENCODER = ModuleCost(luts=60, ffs=20, description="AER spike encoder")
AER_ROUTER = ModuleCost(luts=100, ffs=32, description="AER event router")

# BRAM cost per weight (16-bit Q8.8, stored in BRAM)
BRAM_BITS_PER_WEIGHT = 16
