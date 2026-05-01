# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hardware platform profiles for universal FPGA/ASIC targeting

"""Pre-configured hardware profiles for every target platform.

Each profile encodes the optimal fixed-point configuration for a specific
hardware target, including DSP multiplier widths, overflow handling, and
rounding semantics.

Usage::

    from sc_neurocore.compiler.hardware_profiles import get_profile, list_profiles

    # Compile for Intel Loihi 2
    profile = get_profile("loihi2")
    verilog = neuron.to_verilog(
        module_name="sc_lif",
        data_width=profile.data_width,
        fraction=profile.fraction,
    )

    # List all available targets
    for p in list_profiles():
        print(f"{p.name:16s} {p.vendor:12s} Q{p.int_bits}.{p.fraction} ({p.data_width}-bit)")

Supported Platform Classes
--------------------------
- **Xilinx FPGA** (Spartan-6 through Versal)
- **Intel FPGA** (Cyclone V through Agilex)
- **Lattice** (ECP5, CrossLink-NX, CertusPro-NX)
- **Gowin**, **Efinix**, **Microchip/Microsemi**, **Achronix**, **QuickLogic**
- **Neuromorphic** (Loihi 2, TrueNorth, BrainScaleS-2, SpiNNaker 2, Akida, Dynap)
- **Photonic** (Lightmatter, Xanadu, iPronics, Lightelligence, Luminous)
- **In-Memory / PIM** (UPMEM, Samsung HBM-PIM, SK Hynix AiM, CXL)
- **Superconducting** (NIST SFQ, Northrop AQFP, Josephson)
- **Spintronic** (Everspin STT-MRAM, Samsung SOT-MRAM)
- **Ferroelectric** (GlobalFoundries FeFET, SK Hynix FeRAM)
- **CGRA** (Samsung, Qualcomm NPU, Cadence Xtensa)
- **3D-Stacked** (TSMC SoIC, Intel Foveros, AMD 3D V-Cache)
- **Edge MCU** (RP2040, ESP32-S3, STM32H7, nRF5340, MAX78000)
- **ASIC** (arbitrary standard-cell targets)
- **Simulation** (golden reference)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ── Type aliases ─────────────────────────────────────────────────────
OverflowMode = Literal["saturate", "wrap", "trap"]
RoundingMode = Literal["truncate", "nearest", "bankers", "stochastic"]


@dataclass(frozen=True)
class HardwareProfile:
    """Complete hardware configuration for a target platform.

    Attributes
    ----------
    name : str
        Short machine-readable identifier (e.g. ``"loihi2"``).
    vendor : str
        Chip vendor (e.g. ``"Intel"``, ``"Xilinx"``).
    family : str
        Product family (e.g. ``"Arria 10"``, ``"ECP5"``).
    platform_class : str
        One of ``"fpga"``, ``"neuromorphic"``, ``"asic"``, ``"simulation"``.
    data_width : int
        Total bit width for fixed-point arithmetic.
    fraction : int
        Number of fractional bits.
    signed : bool
        True for signed (two's complement), False for unsigned Q-format.
    overflow : OverflowMode
        How to handle arithmetic overflow in next-state logic.
    rounding : RoundingMode
        How to round after fixed-point multiplication truncation.
    dsp_block : str
        Name of the DSP hard macro (e.g. ``"DSP48E2"``).
    dsp_mult_a : int
        Width of the DSP A-port (multiplier input A).
    dsp_mult_b : int
        Width of the DSP B-port (multiplier input B).
    max_freq_mhz : int
        Typical maximum clock frequency (0 = unknown).
    notes : str
        Human-readable rationale for the configuration.
    """

    name: str
    vendor: str
    family: str
    platform_class: str  # fpga | neuromorphic | asic | simulation
    data_width: int
    fraction: int
    signed: bool = True
    overflow: OverflowMode = "saturate"
    rounding: RoundingMode = "truncate"
    dsp_block: str = ""
    dsp_mult_a: int = 0
    dsp_mult_b: int = 0
    max_freq_mhz: int = 0
    notes: str = ""

    @property
    def int_bits(self) -> int:
        """Number of integer bits (excluding sign bit if signed)."""
        return self.data_width - self.fraction - (1 if self.signed else 0)

    @property
    def q_format_label(self) -> str:
        """Human-readable Q-format string (e.g. ``'Q9.9'`` or ``'UQ8.8'``)."""
        prefix = "Q" if self.signed else "UQ"
        return f"{prefix}{self.int_bits}.{self.fraction}"

    @property
    def max_value(self) -> float:
        """Maximum representable positive value."""
        if self.signed:
            return ((1 << (self.data_width - 1)) - 1) / (1 << self.fraction)
        return ((1 << self.data_width) - 1) / (1 << self.fraction)

    @property
    def min_value(self) -> float:
        """Minimum representable value (most negative or zero)."""
        if self.signed:
            return -(1 << (self.data_width - 1)) / (1 << self.fraction)
        return 0.0

    @property
    def resolution(self) -> float:
        """Smallest representable step."""
        return 1.0 / (1 << self.fraction)

    @classmethod
    def from_constraints(
        cls,
        name: str,
        *,
        vendor: str = "Generic",
        family: str = "Auto",
        platform_class: str = "custom",
        data_width: int | None = None,
        fraction: int | None = None,
        max_freq_mhz: int = 0,
        overflow: OverflowMode = "saturate",
        rounding: RoundingMode = "nearest",
        min_precision_bits: int = 8,
        max_power_budget_mw: float | None = None,
        notes: str = "",
    ) -> "HardwareProfile":
        """Auto-construct an optimal profile from spec-sheet constraints.

        This is the **ultimate extensibility mechanism**: instead of manually
        defining every field, provide constraints and let SC-NeuroCore select
        the optimal fixed-point configuration.

        Parameters
        ----------
        name : str
            Unique profile identifier.
        vendor : str
            Vendor name.
        family : str
            Product family.
        platform_class : str
            Platform class identifier.
        data_width : int, optional
            Override total bit width. Auto-selects if None.
        fraction : int, optional
            Override fraction bits. Auto-selects if None.
        max_freq_mhz : int
            Maximum clock frequency.
        overflow : OverflowMode
            Overflow handling.
        rounding : RoundingMode
            Rounding mode.
        min_precision_bits : int
            Minimum fractional precision required.
        max_power_budget_mw : float, optional
            Power budget constraint (used for width selection).
        notes : str
            Human-readable description.

        Returns
        -------
        HardwareProfile
            Auto-constructed profile.
        """
        # Auto-select data width based on precision and power
        if data_width is None:
            if max_power_budget_mw is not None and max_power_budget_mw < 10:
                data_width = max(8, min_precision_bits)
            elif max_power_budget_mw is not None and max_power_budget_mw < 100:
                data_width = max(16, min_precision_bits * 2)
            else:
                data_width = max(16, min_precision_bits * 2)

        # Auto-select fraction: half the data width, at least min_precision
        if fraction is None:
            fraction = max(min_precision_bits, data_width // 2)
            fraction = min(fraction, data_width - 1)

        profile = cls(
            name=name, vendor=vendor, family=family,
            platform_class=platform_class,
            data_width=data_width, fraction=fraction,
            overflow=overflow, rounding=rounding,
            max_freq_mhz=max_freq_mhz,
            notes=notes or f"Auto-constructed from constraints.",
        )
        # Auto-register
        _PROFILES[name.lower().replace("-", "_").replace(" ", "_")] = profile
        return profile


# ═══════════════════════════════════════════════════════════════════════
# Pre-configured profiles
# ═══════════════════════════════════════════════════════════════════════

_PROFILES: dict[str, HardwareProfile] = {}


def _reg(p: HardwareProfile) -> HardwareProfile:
    """Register a profile in the global registry."""
    _PROFILES[p.name] = p
    return p


# ── Xilinx FPGA ─────────────────────────────────────────────────────

_reg(HardwareProfile(
    name="spartan6", vendor="Xilinx", family="Spartan-6",
    platform_class="fpga", data_width=18, fraction=9,
    dsp_block="DSP48A1", dsp_mult_a=18, dsp_mult_b=18,
    max_freq_mhz=250,
    notes="18-bit DSP48A1 native. Q9.9 uses full B-port.",
))

_reg(HardwareProfile(
    name="artix7", vendor="Xilinx", family="Artix-7",
    platform_class="fpga", data_width=18, fraction=9,
    dsp_block="DSP48E1", dsp_mult_a=25, dsp_mult_b=18,
    max_freq_mhz=450,
    notes="Q9.9 uses full 18-bit B-port. A-port has 7 guard bits.",
))

_reg(HardwareProfile(
    name="kintex7", vendor="Xilinx", family="Kintex-7",
    platform_class="fpga", data_width=18, fraction=9,
    dsp_block="DSP48E1", dsp_mult_a=25, dsp_mult_b=18,
    max_freq_mhz=500,
    notes="Same DSP48E1 as Artix-7 but faster fabric.",
))

_reg(HardwareProfile(
    name="ultrascale", vendor="Xilinx", family="UltraScale",
    platform_class="fpga", data_width=36, fraction=18,
    dsp_block="DSP48E2", dsp_mult_a=27, dsp_mult_b=18,
    max_freq_mhz=600,
    notes="Q18.18 uses full 18-bit B-port × 2 for 36-bit state.",
))

_reg(HardwareProfile(
    name="ultrascale_plus", vendor="Xilinx", family="UltraScale+",
    platform_class="fpga", data_width=36, fraction=18,
    dsp_block="DSP48E2", dsp_mult_a=27, dsp_mult_b=18,
    max_freq_mhz=775,
    notes="Same DSP48E2 architecture, higher speed grade.",
))

_reg(HardwareProfile(
    name="versal", vendor="Xilinx", family="Versal",
    platform_class="fpga", data_width=24, fraction=12,
    dsp_block="DSP58", dsp_mult_a=27, dsp_mult_b=24,
    max_freq_mhz=900,
    notes="DSP58 extends B-port to 24 bits. Q12.12 is native.",
))

# ── Intel FPGA ───────────────────────────────────────────────────────

_reg(HardwareProfile(
    name="cyclone_v", vendor="Intel", family="Cyclone V",
    platform_class="fpga", data_width=18, fraction=9,
    dsp_block="Variable Precision", dsp_mult_a=18, dsp_mult_b=18,
    max_freq_mhz=300,
    notes="18×18 native. Same Q9.9 as Xilinx 7-series.",
))

_reg(HardwareProfile(
    name="cyclone_10", vendor="Intel", family="Cyclone 10",
    platform_class="fpga", data_width=18, fraction=9,
    dsp_block="Variable Precision", dsp_mult_a=18, dsp_mult_b=18,
    max_freq_mhz=300,
    notes="Same DSP as Cyclone V.",
))

_reg(HardwareProfile(
    name="arria10", vendor="Intel", family="Arria 10",
    platform_class="fpga", data_width=27, fraction=13,
    dsp_block="Variable Precision", dsp_mult_a=27, dsp_mult_b=27,
    max_freq_mhz=500,
    notes="27×27 native. Q14.13 uses full multiplier width.",
))

_reg(HardwareProfile(
    name="stratix10", vendor="Intel", family="Stratix 10",
    platform_class="fpga", data_width=27, fraction=13,
    dsp_block="Variable Precision", dsp_mult_a=27, dsp_mult_b=27,
    max_freq_mhz=700,
    notes="Same 27×27 DSP as Arria 10, higher clock.",
))

_reg(HardwareProfile(
    name="agilex", vendor="Intel", family="Agilex",
    platform_class="fpga", data_width=27, fraction=13,
    dsp_block="Variable Precision", dsp_mult_a=27, dsp_mult_b=27,
    max_freq_mhz=800,
    notes="Latest Intel 10nm FPGA. 27×27 native DSP.",
))

# ── Lattice FPGA ─────────────────────────────────────────────────────

_reg(HardwareProfile(
    name="ecp5", vendor="Lattice", family="ECP5",
    platform_class="fpga", data_width=18, fraction=9,
    dsp_block="MULT18X18D", dsp_mult_a=18, dsp_mult_b=18,
    max_freq_mhz=400,
    notes="18×18 native. Popular open-source FPGA (Yosys/nextpnr).",
))

_reg(HardwareProfile(
    name="crosslink_nx", vendor="Lattice", family="CrossLink-NX",
    platform_class="fpga", data_width=18, fraction=9,
    dsp_block="DSP", dsp_mult_a=18, dsp_mult_b=18,
    max_freq_mhz=400,
    notes="28nm, low power. Same 18×18 multiplier as ECP5.",
))

_reg(HardwareProfile(
    name="certuspro_nx", vendor="Lattice", family="CertusPro-NX",
    platform_class="fpga", data_width=18, fraction=9,
    dsp_block="DSP", dsp_mult_a=18, dsp_mult_b=18,
    max_freq_mhz=500,
    notes="Lattice's largest 28nm FPGA. 18×18 DSP.",
))

# ── Other FPGA Vendors ──────────────────────────────────────────────

_reg(HardwareProfile(
    name="gowin", vendor="Gowin", family="GW1N/GW2A",
    platform_class="fpga", data_width=18, fraction=9,
    dsp_block="MULT18X18", dsp_mult_a=18, dsp_mult_b=18,
    max_freq_mhz=300,
    notes="Chinese FPGA. 18×18 native, Yosys-compatible.",
))

_reg(HardwareProfile(
    name="efinix", vendor="Efinix", family="Trion/Titanium",
    platform_class="fpga", data_width=10, fraction=5,
    dsp_block="MULT", dsp_mult_a=10, dsp_mult_b=9,
    max_freq_mhz=300,
    notes="10×9 multiplier. Q5.5 (10-bit) is the native maximum.",
))

_reg(HardwareProfile(
    name="polarfire", vendor="Microchip", family="PolarFire",
    platform_class="fpga", data_width=18, fraction=9,
    dsp_block="MACC", dsp_mult_a=18, dsp_mult_b=18,
    max_freq_mhz=400,
    notes="Radiation-tolerant. 18×18 MACC block.",
))

_reg(HardwareProfile(
    name="smartfusion2", vendor="Microchip", family="SmartFusion2",
    platform_class="fpga", data_width=18, fraction=9,
    dsp_block="MACC", dsp_mult_a=18, dsp_mult_b=18,
    max_freq_mhz=200,
    notes="FPGA + ARM Cortex-M3. 18×18 MACC.",
))

_reg(HardwareProfile(
    name="achronix", vendor="Achronix", family="Speedster7t",
    platform_class="fpga", data_width=24, fraction=12,
    dsp_block="MLP", dsp_mult_a=24, dsp_mult_b=24,
    max_freq_mhz=750,
    notes="Machine Learning Processor block. 24×24 native.",
))

_reg(HardwareProfile(
    name="quicklogic", vendor="QuickLogic", family="EOS S3",
    platform_class="fpga", data_width=8, fraction=4,
    dsp_block="LUT-based", dsp_mult_a=0, dsp_mult_b=0,
    max_freq_mhz=80,
    notes="No DSP blocks. LUT-based multiply. 8-bit optimal.",
))

# ── ICE40 (Lattice, open-source) ────────────────────────────────────

_reg(HardwareProfile(
    name="ice40", vendor="Lattice", family="iCE40",
    platform_class="fpga", data_width=16, fraction=8,
    dsp_block="SB_MAC16", dsp_mult_a=16, dsp_mult_b=16,
    max_freq_mhz=270,
    notes="Tiny open-source FPGA. 16×16 SB_MAC16. Q8.8 native.",
))

# ── Neuromorphic Chips ───────────────────────────────────────────────

_reg(HardwareProfile(
    name="loihi2", vendor="Intel", family="Loihi 2",
    platform_class="neuromorphic", data_width=24, fraction=12,
    overflow="wrap", rounding="truncate",
    notes="24-bit membrane potential. Wrap on overflow (hardware behaviour).",
))

_reg(HardwareProfile(
    name="truenorth", vendor="IBM", family="TrueNorth",
    platform_class="neuromorphic", data_width=8, fraction=7,
    overflow="saturate", rounding="truncate",
    notes="1-bit stochastic neurons. Q1.7 approximation for parameter transfer.",
))

_reg(HardwareProfile(
    name="brainscales2", vendor="Heidelberg", family="BrainScaleS-2",
    platform_class="neuromorphic", data_width=8, fraction=4,
    overflow="saturate", rounding="nearest",
    notes="Analog VLSI with 8-bit DAC/ADC. Q4.4 for digital calibration.",
))

_reg(HardwareProfile(
    name="spinnaker2", vendor="TU Dresden", family="SpiNNaker 2",
    platform_class="neuromorphic", data_width=16, fraction=15,
    overflow="saturate", rounding="nearest",
    notes="ARM Cortex-M4 with CMSIS-DSP. Q1.15 is the native format.",
))

_reg(HardwareProfile(
    name="akida", vendor="BrainChip", family="Akida 2.0",
    platform_class="neuromorphic", data_width=8, fraction=7,
    overflow="saturate", rounding="truncate",
    notes="Event-driven neural processor. 8-bit weights.",
))

_reg(HardwareProfile(
    name="dynap_se2", vendor="SynSense", family="DYNAP-SE2",
    platform_class="neuromorphic", data_width=16, fraction=8,
    overflow="saturate", rounding="truncate",
    notes="Mixed-signal neuromorphic. 16-bit digital membrane.",
))

_reg(HardwareProfile(
    name="xylo", vendor="SynSense", family="Xylo",
    platform_class="neuromorphic", data_width=16, fraction=8,
    overflow="saturate", rounding="truncate",
    notes="Digital spiking neural network processor. 16-bit.",
))

# ── ASIC Targets ─────────────────────────────────────────────────────

_reg(HardwareProfile(
    name="asic_16", vendor="ASIC", family="Standard Cell (16-bit)",
    platform_class="asic", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    notes="Generic 16-bit ASIC. No DSP constraint — any width is synthesisable.",
))

_reg(HardwareProfile(
    name="asic_32", vendor="ASIC", family="Standard Cell (32-bit)",
    platform_class="asic", data_width=32, fraction=16,
    overflow="saturate", rounding="nearest",
    notes="Generic 32-bit ASIC. Q16.16 gold standard.",
))

_reg(HardwareProfile(
    name="asic_custom", vendor="ASIC", family="Custom",
    platform_class="asic", data_width=24, fraction=12,
    overflow="trap", rounding="bankers",
    notes="Safety-critical ASIC (DO-254/IEC 61508). Trap on overflow.",
))

# ── Simulation / Golden Reference ────────────────────────────────────

_reg(HardwareProfile(
    name="sim_q88", vendor="Simulation", family="Icarus Q8.8",
    platform_class="simulation", data_width=16, fraction=8,
    notes="Default simulation target for iverilog co-simulation.",
))

_reg(HardwareProfile(
    name="sim_q1616", vendor="Simulation", family="Icarus Q16.16",
    platform_class="simulation", data_width=32, fraction=16,
    notes="Gold standard simulation for fidelity validation.",
))

# ── Additional FPGA (2025–2026 platforms) ────────────────────────────

_reg(HardwareProfile(
    name="alveo", vendor="AMD/Xilinx", family="Alveo U50/U200/U280",
    platform_class="fpga", data_width=36, fraction=18,
    dsp_block="DSP48E2", dsp_mult_a=27, dsp_mult_b=18,
    max_freq_mhz=700,
    notes="Data-centre accelerator card. Same DSP48E2 as UltraScale+.",
))

_reg(HardwareProfile(
    name="nexus", vendor="Lattice", family="Nexus (LIFCL)",
    platform_class="fpga", data_width=18, fraction=9,
    dsp_block="DSP", dsp_mult_a=18, dsp_mult_b=18,
    max_freq_mhz=500,
    notes="28nm replacement for ECP5. 18×18 DSP.",
))

_reg(HardwareProfile(
    name="polarfire_soc", vendor="Microchip", family="PolarFire SoC",
    platform_class="fpga", data_width=18, fraction=9,
    dsp_block="MACC", dsp_mult_a=18, dsp_mult_b=18,
    max_freq_mhz=400,
    notes="FPGA + RISC-V hard core. 18×18 MACC. ARM-free.",
))

_reg(HardwareProfile(
    name="avant", vendor="Lattice", family="Avant",
    platform_class="fpga", data_width=18, fraction=9,
    dsp_block="DSP", dsp_mult_a=18, dsp_mult_b=18,
    max_freq_mhz=500,
    notes="Next-gen Lattice 16nm mid-range FPGA.",
))

# ── AI / ML Accelerators ────────────────────────────────────────────

_reg(HardwareProfile(
    name="tpu", vendor="Google", family="TPU v4/v5",
    platform_class="accelerator", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    notes="Tensor Processing Unit. INT8/bfloat16. Q8.8 for param transfer.",
))

_reg(HardwareProfile(
    name="cerebras_wse", vendor="Cerebras", family="WSE-3",
    platform_class="accelerator", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    notes="Wafer-scale engine. 900K cores, FP16 native.",
))

_reg(HardwareProfile(
    name="graphcore_ipu", vendor="Graphcore", family="IPU Mk2/Bow",
    platform_class="accelerator", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    notes="Intelligence Processing Unit. FP16 accumulation.",
))

_reg(HardwareProfile(
    name="tenstorrent", vendor="Tenstorrent", family="Grayskull/Wormhole",
    platform_class="accelerator", data_width=16, fraction=8,
    overflow="saturate", rounding="truncate",
    notes="RISC-V based AI accelerator. 8/16-bit tensor ops.",
))

_reg(HardwareProfile(
    name="ethos_u", vendor="ARM", family="Ethos-U55/U65",
    platform_class="accelerator", data_width=8, fraction=4,
    overflow="saturate", rounding="nearest",
    notes="Micro NPU for Cortex-M edge AI. 8-bit native.",
))

_reg(HardwareProfile(
    name="hexagon", vendor="Qualcomm", family="Hexagon DSP/HVX",
    platform_class="accelerator", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    notes="Mobile DSP with 16-bit HVX SIMD.",
))

_reg(HardwareProfile(
    name="apple_ane", vendor="Apple", family="Neural Engine (M-series)",
    platform_class="accelerator", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    notes="Apple Neural Engine. 16-bit inference.",
))

# ── DSP Processors ──────────────────────────────────────────────────

_reg(HardwareProfile(
    name="sharc", vendor="Analog Devices", family="SHARC ADSP-SC5xx",
    platform_class="dsp", data_width=32, fraction=16,
    overflow="saturate", rounding="nearest",
    notes="32/40-bit fixed-point DSP. Q16.16 native.",
))

_reg(HardwareProfile(
    name="c6000", vendor="Texas Instruments", family="C66x/C67x",
    platform_class="dsp", data_width=32, fraction=16,
    overflow="saturate", rounding="nearest",
    notes="40-bit accumulator. Q16.16 for fixed-point mode.",
))

_reg(HardwareProfile(
    name="ceva_xc", vendor="CEVA", family="CEVA-XC/X2",
    platform_class="dsp", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    notes="Communications DSP. 16-bit Q8.8 for real-time.",
))

# ── Emerging Compute Paradigms ───────────────────────────────────────

_reg(HardwareProfile(
    name="photonic", vendor="Lightmatter", family="Envise/Passage",
    platform_class="emerging", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="Optical matrix multiply. 8-bit DAC/ADC resolution.",
))

_reg(HardwareProfile(
    name="riscv_fpga", vendor="SiFive", family="RISC-V + FPGA fabric",
    platform_class="emerging", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    notes="Open ISA with custom neural extensions. Q8.8 baseline.",
))

_reg(HardwareProfile(
    name="in_memory", vendor="Mythic/Syntiant", family="Analog In-Memory",
    platform_class="emerging", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="Computation in flash/SRAM. 4-8 bit effective resolution.",
))

_reg(HardwareProfile(
    name="quantum_hybrid", vendor="IBM/Custom", family="Quantum-Classical",
    platform_class="emerging", data_width=32, fraction=16,
    overflow="saturate", rounding="bankers",
    notes="Quantum-classical interface. Q16.16 for parameter encoding.",
))

# ── Radiation-Hardened / Space ───────────────────────────────────────

_reg(HardwareProfile(
    name="nanoxplore", vendor="NanoXplore", family="NG-Ultra",
    platform_class="fpga", data_width=18, fraction=9,
    dsp_block="DSP", dsp_mult_a=18, dsp_mult_b=18,
    overflow="trap", rounding="nearest",
    max_freq_mhz=300,
    notes="European rad-hard FPGA (ESA approved). SEU-tolerant. 18×18 DSP.",
))

_reg(HardwareProfile(
    name="rtg4", vendor="Microchip", family="RTG4",
    platform_class="fpga", data_width=18, fraction=9,
    dsp_block="MACC", dsp_mult_a=18, dsp_mult_b=18,
    overflow="trap", rounding="nearest",
    max_freq_mhz=200,
    notes="Rad-tolerant flash FPGA. NASA/ESA heritage. 18×18 MACC.",
))

_reg(HardwareProfile(
    name="kintex_us_rt", vendor="Xilinx", family="Kintex UltraScale RT",
    platform_class="fpga", data_width=36, fraction=18,
    dsp_block="DSP48E2", dsp_mult_a=27, dsp_mult_b=18,
    overflow="trap", rounding="nearest",
    max_freq_mhz=500,
    notes="Space-grade Kintex (XQRKU060). TMR-compatible. 27×18 DSP.",
))

# ── Edge AI Accelerators ────────────────────────────────────────────

_reg(HardwareProfile(
    name="hailo8", vendor="Hailo", family="Hailo-8/8L",
    platform_class="accelerator", data_width=8, fraction=4,
    overflow="saturate", rounding="nearest",
    notes="26 TOPS edge AI. Dominant in automotive ADAS.",
))

_reg(HardwareProfile(
    name="kneron", vendor="Kneron", family="KL730",
    platform_class="accelerator", data_width=8, fraction=4,
    overflow="saturate", rounding="nearest",
    notes="Ultra-low-power edge NPU. 4-bit/8-bit modes.",
))

_reg(HardwareProfile(
    name="groq_tsp", vendor="Groq", family="TSP (LPU)",
    platform_class="accelerator", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    notes="Tensor Streaming Processor. Deterministic latency, no caches.",
))

_reg(HardwareProfile(
    name="jetson", vendor="NVIDIA", family="Jetson Orin",
    platform_class="accelerator", data_width=8, fraction=4,
    overflow="saturate", rounding="nearest",
    notes="INT8 tensor cores. Most deployed edge GPU platform.",
))

_reg(HardwareProfile(
    name="habana_gaudi", vendor="Intel", family="Habana Gaudi 2/3",
    platform_class="accelerator", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    notes="Data centre AI training. FP8/INT8/BF16 native.",
))

_reg(HardwareProfile(
    name="drp_ai", vendor="Renesas", family="RZ/V2H DRP-AI",
    platform_class="accelerator", data_width=8, fraction=4,
    overflow="saturate", rounding="nearest",
    notes="Automotive AI accelerator. INT8 inference.",
))

# ── Embedded FPGA (eFPGA) IP ────────────────────────────────────────

_reg(HardwareProfile(
    name="speedcore", vendor="Achronix", family="Speedcore eFPGA IP",
    platform_class="fpga", data_width=24, fraction=12,
    dsp_block="MLP", dsp_mult_a=24, dsp_mult_b=24,
    max_freq_mhz=750,
    notes="eFPGA IP embedded in custom ASICs. 24×24 MLP blocks.",
))

_reg(HardwareProfile(
    name="eflx", vendor="Flex Logix", family="EFLX eFPGA",
    platform_class="fpga", data_width=16, fraction=8,
    dsp_block="LUT-based", dsp_mult_a=0, dsp_mult_b=0,
    max_freq_mhz=500,
    notes="eFPGA for post-silicon reconfigurability. LUT-based multiply.",
))

_reg(HardwareProfile(
    name="menta_efpga", vendor="Menta", family="Origami eFPGA",
    platform_class="fpga", data_width=16, fraction=8,
    dsp_block="LUT-based", dsp_mult_a=0, dsp_mult_b=0,
    max_freq_mhz=400,
    notes="European eFPGA IP. LUT-based arithmetic.",
))

# ── Vision / Sensor Processors ──────────────────────────────────────

_reg(HardwareProfile(
    name="imx500", vendor="Sony", family="IMX500/IMX501",
    platform_class="accelerator", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="AI-on-sensor. Processes spikes at the pixel level.",
))

_reg(HardwareProfile(
    name="samsung_npu", vendor="Samsung", family="Exynos NPU",
    platform_class="accelerator", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    notes="Exynos NPU in billions of phones. INT8/INT16.",
))

# ── Neuromorphic: Next-Generation (2025-2026) ───────────────────────
_reg(HardwareProfile(
    name="loihi3", vendor="Intel", family="Loihi 3",
    platform_class="neuromorphic", data_width=32, fraction=16,
    overflow="wrap", rounding="truncate",
    notes="Loihi 3 (4nm, 8M neurons). 32-bit state, wrap overflow.",
))
_reg(HardwareProfile(
    name="northpole", vendor="IBM", family="NorthPole",
    platform_class="neuromorphic", data_width=8, fraction=4,
    overflow="saturate", rounding="nearest",
    notes="IBM NorthPole: 256-core digital, no DRAM. INT2/INT4/INT8.",
))
_reg(HardwareProfile(
    name="innatera_pulsar", vendor="Innatera", family="Pulsar",
    platform_class="neuromorphic", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="Innatera Pulsar neuromorphic μC. Analog-digital hybrid.",
))

# ── FPGA: Missing Families ──────────────────────────────────────────
_reg(HardwareProfile(
    name="versal_ai_edge", vendor="AMD/Xilinx", family="Versal AI Edge",
    platform_class="fpga", data_width=16, fraction=8,
    dsp_block="DSP58", dsp_mult_a=27, dsp_mult_b=24,
    max_freq_mhz=900, overflow="saturate", rounding="nearest",
    notes="Versal AI Edge: AI Engines + PL. DSP58 = 27x24.",
))
_reg(HardwareProfile(
    name="proasic3", vendor="Microchip", family="ProASIC3",
    platform_class="fpga", data_width=16, fraction=8,
    overflow="saturate", rounding="truncate",
    notes="Flash-based, no bitstream volatility. Aerospace legacy.",
))
_reg(HardwareProfile(
    name="trion", vendor="Efinix", family="Trion T-series",
    platform_class="fpga", data_width=16, fraction=8,
    dsp_block="MULT", dsp_mult_a=10, dsp_mult_b=10,
    max_freq_mhz=200, overflow="saturate", rounding="truncate",
    notes="Efinix Trion: ultra-low-cost Quantum fabric. 10x10 mult.",
))
_reg(HardwareProfile(
    name="titanium", vendor="Efinix", family="Titanium Ti-series",
    platform_class="fpga", data_width=18, fraction=9,
    dsp_block="MULT", dsp_mult_a=18, dsp_mult_b=18,
    max_freq_mhz=500, overflow="saturate", rounding="nearest",
    notes="Efinix Titanium: hardened RISC-V core + 18x18 mult.",
))
_reg(HardwareProfile(
    name="gowin_arora_v", vendor="Gowin", family="Arora V GW5A",
    platform_class="fpga", data_width=18, fraction=9,
    dsp_block="DSP", dsp_mult_a=18, dsp_mult_b=18,
    max_freq_mhz=400, overflow="saturate", rounding="truncate",
    notes="Gowin Arora V 28nm: premium tier with 18x18 DSP.",
))
_reg(HardwareProfile(
    name="intel_agilex5", vendor="Intel", family="Agilex 5 E/D",
    platform_class="fpga", data_width=18, fraction=9,
    dsp_block="DSP", dsp_mult_a=18, dsp_mult_b=19,
    max_freq_mhz=800, overflow="saturate", rounding="nearest",
    notes="Intel Agilex 5 E/D-series, HBM2e. Latest mid-range.",
))

# ── AI Accelerators: Major Missing ──────────────────────────────────
_reg(HardwareProfile(
    name="nvidia_dla", vendor="NVIDIA", family="DLA (Orin)",
    platform_class="accelerator", data_width=8, fraction=4,
    overflow="saturate", rounding="nearest",
    notes="NVIDIA DLA: dedicated INT8 inference engine in Orin SoC.",
))
_reg(HardwareProfile(
    name="mediatek_apu", vendor="MediaTek", family="APU 790",
    platform_class="accelerator", data_width=8, fraction=4,
    overflow="saturate", rounding="nearest",
    notes="MediaTek APU: ~40% mobile AI market. INT4/INT8/INT16.",
))
_reg(HardwareProfile(
    name="aws_inferentia", vendor="AWS", family="Inferentia2",
    platform_class="accelerator", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    notes="AWS Inferentia2/Trainium2. BF16/FP16/INT8 cloud inference.",
))
_reg(HardwareProfile(
    name="qualcomm_nsp", vendor="Qualcomm", family="Neural Signal Processor",
    platform_class="accelerator", data_width=8, fraction=4,
    overflow="saturate", rounding="nearest",
    notes="Qualcomm NSP: dedicated INT8 inference engine (not HVX DSP).",
))
_reg(HardwareProfile(
    name="sambanova", vendor="SambaNova", family="RDU",
    platform_class="accelerator", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    notes="SambaNova RDU: reconfigurable dataflow architecture. BF16/FP16.",
))
_reg(HardwareProfile(
    name="cambricon_mlu", vendor="Cambricon", family="MLU370",
    platform_class="accelerator", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    notes="Cambricon MLU370/590: dominant Chinese AI silicon. INT8/FP16.",
))

# ── Emerging Compute Paradigms ──────────────────────────────────────
_reg(HardwareProfile(
    name="superconducting", vendor="Various", family="AQFP/SFQ",
    platform_class="emerging", data_width=8, fraction=4,
    overflow="wrap", rounding="truncate",
    notes="Superconducting AQFP/SFQ: ~100 GHz, μW power. Post-CMOS.",
))
_reg(HardwareProfile(
    name="cim_sram", vendor="Various", family="CIM-SRAM",
    platform_class="emerging", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="Compute-in-SRAM (TSMC/Samsung): analog MAC in memory cell.",
))
_reg(HardwareProfile(
    name="analog_ai", vendor="IBM/Mythic", family="PCM/ReRAM",
    platform_class="emerging", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="Phase-change / ReRAM analog compute: 0.1 TOPS/W class.",
))
_reg(HardwareProfile(
    name="event_camera", vendor="Prophesee/Sony", family="DVS/IMX636",
    platform_class="emerging", data_width=16, fraction=8,
    overflow="saturate", rounding="truncate",
    notes="DVS event-camera: natural AER spike interface. IMX636 native.",
))

# ── Photonic / Optical Compute ──────────────────────────────────────
_reg(HardwareProfile(
    name="lightmatter_passage", vendor="Lightmatter", family="Passage",
    platform_class="photonic", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    dsp_block="MZI", dsp_mult_a=8, dsp_mult_b=8,
    notes="Lightmatter Passage: photonic MZI matrix multiply. Phase-shift weights.",
))
_reg(HardwareProfile(
    name="lightelligence_pace", vendor="Lightelligence", family="PACE2",
    platform_class="photonic", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    dsp_block="OE-MAC", dsp_mult_a=8, dsp_mult_b=8,
    notes="Lightelligence PACE2: hybrid opto-electronic accelerator. IPO 2026.",
))
_reg(HardwareProfile(
    name="xanadu_x8", vendor="Xanadu", family="X8/Borealis",
    platform_class="photonic", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    dsp_block="GBS", dsp_mult_a=8, dsp_mult_b=8,
    notes="Xanadu X8/Borealis: photonic quantum-classical hybrid.",
))
_reg(HardwareProfile(
    name="ipronics_smartlight", vendor="iPronics", family="SmartLight",
    platform_class="photonic", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    dsp_block="MZI", dsp_mult_a=8, dsp_mult_b=8,
    notes="iPronics SmartLight: programmable photonic processor. EU-funded.",
))
_reg(HardwareProfile(
    name="luminous_computing", vendor="Luminous", family="Photonic Engine",
    platform_class="photonic", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    dsp_block="MZI", dsp_mult_a=8, dsp_mult_b=8,
    notes="Luminous Computing: optical interconnect + compute co-design.",
))

# ── Chiplet / UCIe / Heterogeneous Integration ──────────────────────
_reg(HardwareProfile(
    name="tenstorrent_blackhole", vendor="Tenstorrent", family="Blackhole",
    platform_class="accelerator", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    dsp_block="Tensix", dsp_mult_a=16, dsp_mult_b=16,
    notes="Tenstorrent Blackhole: RISC-V + AI chiplet. Galaxy platform 2026.",
))
_reg(HardwareProfile(
    name="cerebras_wse3", vendor="Cerebras", family="WSE-3",
    platform_class="accelerator", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    dsp_block="PE", dsp_mult_a=16, dsp_mult_b=16,
    max_freq_mhz=1000,
    notes="Cerebras WSE-3: wafer-scale engine. $10B+ OpenAI contract.",
))
_reg(HardwareProfile(
    name="intel_ponte_vecchio", vendor="Intel", family="Ponte Vecchio",
    platform_class="accelerator", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    dsp_block="XMX", dsp_mult_a=16, dsp_mult_b=16,
    notes="Intel Ponte Vecchio: multi-tile GPU, UCIe interconnect.",
))
_reg(HardwareProfile(
    name="amd_mi300x", vendor="AMD", family="Instinct MI300X",
    platform_class="accelerator", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    dsp_block="CDNA3", dsp_mult_a=16, dsp_mult_b=16,
    notes="AMD MI300X: hybrid CPU+GPU+HBM chiplet. 3D-stacked.",
))
_reg(HardwareProfile(
    name="ucie_generic", vendor="UCIe Consortium", family="UCIe 1.1",
    platform_class="accelerator", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    notes="Universal Chiplet Interconnect Express generic template.",
))

# ── Processing-in-Memory / CXL ──────────────────────────────────────
_reg(HardwareProfile(
    name="upmem_pim", vendor="UPMEM", family="PIM-DRAM",
    platform_class="in_memory", data_width=32, fraction=16,
    overflow="saturate", rounding="truncate",
    notes="UPMEM PIM-DRAM: 2560 DPUs per module. In-DRAM processing.",
))
_reg(HardwareProfile(
    name="samsung_hbm_pim", vendor="Samsung", family="HBM-PIM",
    platform_class="in_memory", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    notes="Samsung HBM-PIM: in-bank compute for attention/embedding.",
))
_reg(HardwareProfile(
    name="sk_hynix_aim", vendor="SK Hynix", family="AIM",
    platform_class="in_memory", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    notes="SK Hynix Accelerator-in-Memory: GDDR6-AiM for CXL.",
))
_reg(HardwareProfile(
    name="cxl_type3", vendor="CXL Consortium", family="CXL 3.0 Type-3",
    platform_class="in_memory", data_width=32, fraction=16,
    overflow="saturate", rounding="nearest",
    notes="CXL 3.0 Type-3 memory expander: shared memory pooling.",
))
_reg(HardwareProfile(
    name="axdimm", vendor="Samsung", family="AxDIMM",
    platform_class="in_memory", data_width=16, fraction=8,
    overflow="saturate", rounding="truncate",
    notes="Samsung AxDIMM: near-DRAM processing for embedding lookups.",
))

# ── Next-Gen Neuromorphic ───────────────────────────────────────────
_reg(HardwareProfile(
    name="akida2", vendor="BrainChip", family="Akida 2 / AKD1500",
    platform_class="neuromorphic", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="BrainChip Akida 2: event-based, 1/4/8-bit quant, on-chip learning.",
))
_reg(HardwareProfile(
    name="spinnaker2", vendor="SpiNNcloud", family="SpiNNaker 2",
    platform_class="neuromorphic", data_width=32, fraction=16,
    overflow="wrap", rounding="truncate",
    notes="SpiNNaker 2: ARM-based massively parallel GALS. Brain-scale sim.",
))
_reg(HardwareProfile(
    name="dynapse2", vendor="SynSense", family="DYNAP-SE2",
    platform_class="neuromorphic", data_width=16, fraction=8,
    overflow="saturate", rounding="truncate",
    notes="SynSense DYNAP-SE2: mixed-signal analog neuron circuits.",
))
_reg(HardwareProfile(
    name="rain_neuromorphic", vendor="Rain AI", family="Rain NeuralCore",
    platform_class="neuromorphic", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="Rain AI: memristive crossbar architecture. $100M+ funded.",
))
_reg(HardwareProfile(
    name="brainscales2", vendor="Heidelberg", family="BrainScaleS-2",
    platform_class="neuromorphic", data_width=8, fraction=4,
    overflow="wrap", rounding="truncate",
    notes="BrainScaleS-2: analog accelerated neuro, 1000× bio realtime.",
))

# ── Sovereign / Defence / Aerospace ─────────────────────────────────
_reg(HardwareProfile(
    name="bae_rad750", vendor="BAE Systems", family="RAD750",
    platform_class="fpga", data_width=32, fraction=16,
    overflow="saturate", rounding="truncate",
    dsp_block="", dsp_mult_a=0, dsp_mult_b=0,
    notes="BAE RAD750: rad-hard PowerPC. Mars rovers, deep space.",
))
_reg(HardwareProfile(
    name="cobham_ut700", vendor="Cobham/CAES", family="UT700 LEON3",
    platform_class="fpga", data_width=32, fraction=16,
    overflow="saturate", rounding="truncate",
    dsp_block="", dsp_mult_a=0, dsp_mult_b=0,
    notes="Cobham UT700: LEON3 SPARC rad-hard. ESA space missions.",
))
_reg(HardwareProfile(
    name="mpfs250t_rt", vendor="Microchip", family="PolarFire SoC RT",
    platform_class="fpga", data_width=24, fraction=12,
    overflow="saturate", rounding="nearest",
    dsp_block="MACC", dsp_mult_a=18, dsp_mult_b=18,
    max_freq_mhz=250,
    notes="PolarFire SoC RT: rad-tolerant RISC-V + FPGA. LEO satellites.",
))
_reg(HardwareProfile(
    name="versal_xqrvc1902", vendor="AMD/Xilinx", family="Versal RT",
    platform_class="fpga", data_width=32, fraction=16,
    overflow="saturate", rounding="nearest",
    dsp_block="DSP58", dsp_mult_a=27, dsp_mult_b=24,
    max_freq_mhz=600,
    notes="AMD Versal XQRVC1902: rad-hard Versal AI Core for satellites.",
))
_reg(HardwareProfile(
    name="trenz_zynq_space", vendor="Trenz", family="Zynq Space-Grade",
    platform_class="fpga", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    dsp_block="DSP48E1", dsp_mult_a=25, dsp_mult_b=18,
    max_freq_mhz=300,
    notes="Trenz space-grade Zynq module: ESA-qualified for LEO/GEO.",
))

# ── Automotive / Edge AI SoC ────────────────────────────────────────
_reg(HardwareProfile(
    name="mythic_m1076", vendor="Mythic", family="M1076",
    platform_class="accelerator", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="Mythic M1076: analog CIM. Honda automotive partnership 2026.",
))
_reg(HardwareProfile(
    name="mobileye_eyeq6", vendor="Mobileye", family="EyeQ6",
    platform_class="accelerator", data_width=8, fraction=4,
    overflow="saturate", rounding="nearest",
    dsp_block="VMP", dsp_mult_a=8, dsp_mult_b=8,
    notes="Mobileye EyeQ6: 90 TOPS automotive vision SoC. INT8.",
))
_reg(HardwareProfile(
    name="horizon_j6", vendor="Horizon Robotics", family="Journey 6",
    platform_class="accelerator", data_width=8, fraction=4,
    overflow="saturate", rounding="nearest",
    dsp_block="BPU", dsp_mult_a=8, dsp_mult_b=8,
    notes="Horizon J6: BPU neural accelerator. Chinese auto market leader.",
))
_reg(HardwareProfile(
    name="ambarella_cv72s", vendor="Ambarella", family="CV72S",
    platform_class="accelerator", data_width=8, fraction=4,
    overflow="saturate", rounding="nearest",
    dsp_block="CVflow", dsp_mult_a=8, dsp_mult_b=8,
    notes="Ambarella CV72S: edge AI SoC. Automotive + security cameras.",
))
_reg(HardwareProfile(
    name="hailo15", vendor="Hailo", family="Hailo-15",
    platform_class="accelerator", data_width=8, fraction=4,
    overflow="saturate", rounding="nearest",
    dsp_block="Structure", dsp_mult_a=8, dsp_mult_b=8,
    notes="Hailo-15: next-gen edge AI. 20 TOPS/W, vision processor.",
))
_reg(HardwareProfile(
    name="syntiant_ndp120", vendor="Syntiant", family="NDP120",
    platform_class="neuromorphic", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="Syntiant NDP120: ultra-low-power NDP, always-on audio/sensor.",
))

# ── Superconducting / Cryogenic ──────────────────────────────────────

_reg(HardwareProfile(
    name="nist_sfq", vendor="NIST", family="SFQ",
    platform_class="superconducting", data_width=8, fraction=4,
    overflow="wrap", rounding="truncate",
    max_freq_mhz=100000,  # 100 GHz SFQ
    notes="NIST SFQ: Single Flux Quantum logic. 100 GHz, µW at 4K.",
))
_reg(HardwareProfile(
    name="northrop_aqfp", vendor="Northrop Grumman", family="AQFP",
    platform_class="superconducting", data_width=8, fraction=4,
    overflow="wrap", rounding="truncate",
    max_freq_mhz=5000,  # 5 GHz AQFP
    notes="Northrop Grumman AQFP: Adiabatic QFP. 5 GHz, near-zero power.",
))
_reg(HardwareProfile(
    name="josephson_jj", vendor="Research", family="Josephson",
    platform_class="superconducting", data_width=8, fraction=4,
    overflow="wrap", rounding="truncate",
    max_freq_mhz=50000,
    notes="Josephson Junction neurons: superconducting neuron analogue.",
))

# ── Spintronic / MRAM ────────────────────────────────────────────────

_reg(HardwareProfile(
    name="everspin_stt_mram", vendor="Everspin", family="STT-MRAM",
    platform_class="spintronic", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="Everspin STT-MRAM: non-volatile synaptic weight store. Zero standby.",
))
_reg(HardwareProfile(
    name="samsung_sot_mram", vendor="Samsung", family="SOT-MRAM",
    platform_class="spintronic", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="Samsung SOT-MRAM: spin-orbit torque. Faster write, lower endurance.",
))

# ── Ferroelectric CiM ───────────────────────────────────────────────

_reg(HardwareProfile(
    name="gf_fefet", vendor="GlobalFoundries", family="FeFET-22FDX",
    platform_class="ferroelectric", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="GF 22FDX FeFET: HfO₂ ferroelectric compute-in-memory. MLC capable.",
))
_reg(HardwareProfile(
    name="sk_hynix_feram", vendor="SK Hynix", family="FeRAM",
    platform_class="ferroelectric", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="SK Hynix FeRAM: ferroelectric RAM for non-volatile neural state.",
))

# ── CGRA / Coarse-Grained Reconfigurable ─────────────────────────────

_reg(HardwareProfile(
    name="samsung_cgra", vendor="Samsung", family="CGRA-NPU",
    platform_class="cgra", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    dsp_block="PE", dsp_mult_a=16, dsp_mult_b=16,
    max_freq_mhz=1000,
    notes="Samsung Exynos CGRA: reconfigurable PE array for on-device AI.",
))
_reg(HardwareProfile(
    name="qualcomm_npu_cgra", vendor="Qualcomm", family="NPU-CGRA",
    platform_class="cgra", data_width=8, fraction=4,
    overflow="saturate", rounding="nearest",
    dsp_block="NPU-PE", dsp_mult_a=8, dsp_mult_b=8,
    max_freq_mhz=1200,
    notes="Qualcomm Hexagon NPU CGRA: reconfigurable accelerator in Snapdragon.",
))
_reg(HardwareProfile(
    name="pact_xtensa", vendor="Cadence", family="Xtensa-CGRA",
    platform_class="cgra", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    dsp_block="TIE", dsp_mult_a=16, dsp_mult_b=16,
    max_freq_mhz=800,
    notes="Cadence Xtensa CGRA: configurable processor with CGRA extension.",
))

# ── 3D-Stacked / Monolithic 3D ──────────────────────────────────────

_reg(HardwareProfile(
    name="tsmc_soic", vendor="TSMC", family="SoIC-3D",
    platform_class="3d_stacked", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    max_freq_mhz=2000,
    notes="TSMC SoIC: monolithic 3D. Vertical neuron-synapse partitioning.",
))
_reg(HardwareProfile(
    name="intel_foveros", vendor="Intel", family="Foveros-Direct",
    platform_class="3d_stacked", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    max_freq_mhz=1800,
    notes="Intel Foveros Direct: 3D face-to-face bonding for heterogeneous AI.",
))
_reg(HardwareProfile(
    name="amd_3dv", vendor="AMD", family="3D V-Cache",
    platform_class="3d_stacked", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    max_freq_mhz=2200,
    notes="AMD 3D V-Cache: stacked SRAM for massive synaptic weight cache.",
))

# ── Edge MCU / TinyML ────────────────────────────────────────────────

_reg(HardwareProfile(
    name="rp2040", vendor="Raspberry Pi", family="RP2040",
    platform_class="edge_mcu", data_width=16, fraction=8,
    overflow="wrap", rounding="truncate",
    max_freq_mhz=133,
    notes="RP2040: dual Cortex-M0+, 264KB SRAM. TinyML SNN inference.",
))
_reg(HardwareProfile(
    name="esp32_s3", vendor="Espressif", family="ESP32-S3",
    platform_class="edge_mcu", data_width=16, fraction=8,
    overflow="wrap", rounding="truncate",
    max_freq_mhz=240,
    notes="ESP32-S3: dual-core Xtensa + vector extensions. WiFi/BLE SNN.",
))
_reg(HardwareProfile(
    name="stm32h7", vendor="STMicroelectronics", family="STM32H7",
    platform_class="edge_mcu", data_width=32, fraction=16,
    overflow="saturate", rounding="truncate",
    max_freq_mhz=480,
    notes="STM32H7: Cortex-M7 @ 480 MHz. Best-in-class MCU for SNN.",
))
_reg(HardwareProfile(
    name="nrf5340", vendor="Nordic", family="nRF5340",
    platform_class="edge_mcu", data_width=16, fraction=8,
    overflow="wrap", rounding="truncate",
    max_freq_mhz=128,
    notes="nRF5340: dual Cortex-M33. BLE 5.3 + edge SNN inference.",
))
_reg(HardwareProfile(
    name="max78000", vendor="Analog Devices", family="MAX78000",
    platform_class="edge_mcu", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    dsp_block="CNN", dsp_mult_a=8, dsp_mult_b=8,
    max_freq_mhz=100,
    notes="MAX78000: integrated CNN accelerator. Ultra-low-power edge AI.",
))

# ── RISC-V AI Accelerators ──────────────────────────────────────────

_reg(HardwareProfile(
    name="sifive_x280", vendor="SiFive", family="Intelligence-X280",
    platform_class="accelerator", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    dsp_block="VALU", dsp_mult_a=16, dsp_mult_b=16,
    max_freq_mhz=2000,
    notes="SiFive X280: RISC-V vector processor. RVV 1.0 for SNN kernels.",
))
_reg(HardwareProfile(
    name="qualcomm_ventana", vendor="Qualcomm/Ventana", family="Veyron-V2",
    platform_class="accelerator", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    max_freq_mhz=3000,
    notes="Qualcomm/Ventana Veyron: high-performance RISC-V for data center AI.",
))
_reg(HardwareProfile(
    name="ainekko_rv", vendor="Ainekko", family="ET-Mirai",
    platform_class="accelerator", data_width=8, fraction=4,
    overflow="saturate", rounding="nearest",
    dsp_block="MINION", dsp_mult_a=8, dsp_mult_b=8,
    max_freq_mhz=1000,
    notes="Ainekko ET-Mirai: open-source many-core RISC-V (ex-Esperanto).",
))

# ── Biological / Wetware ─────────────────────────────────────────────

_reg(HardwareProfile(
    name="finalspark_neuroplatform", vendor="FinalSpark",
    family="Neuroplatform",
    platform_class="biological", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    notes="FinalSpark Neuroplatform: living organoid co-processor. "
          "Output = electrode stimulation protocol.",
))
_reg(HardwareProfile(
    name="cortical_labs_dishbrain", vendor="Cortical Labs",
    family="DishBrain",
    platform_class="biological", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    notes="Cortical Labs DishBrain: in-vitro biological neural network. "
          "Output = MEA stimulation pattern.",
))

# ── Electrochemical / Memristive ─────────────────────────────────────

_reg(HardwareProfile(
    name="ibm_ecram", vendor="IBM", family="ECRAM-AnalogAI",
    platform_class="electrochemical", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="IBM ECRAM: electrochemical RAM. Superior linearity for on-chip "
          "learning. Multi-level analog weights.",
))
_reg(HardwareProfile(
    name="samsung_pcram", vendor="Samsung", family="PCRAM",
    platform_class="electrochemical", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="Samsung PCRAM: phase-change memory compute. Non-volatile, "
          "multi-level cell analog synapses.",
))
_reg(HardwareProfile(
    name="stanford_ecram", vendor="Stanford", family="ECRAM-Research",
    platform_class="electrochemical", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="Stanford ECRAM: research-grade electrochemical synapse array. "
          "WO₃-based, 10⁶ endurance cycles.",
))

# ── Wafer-Scale ──────────────────────────────────────────────────────

_reg(HardwareProfile(
    name="cerebras_wse3_ws", vendor="Cerebras", family="WSE-3-WS",
    platform_class="wafer_scale", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    dsp_block="PE", dsp_mult_a=16, dsp_mult_b=16,
    max_freq_mhz=1000,
    notes="Cerebras WSE-3 wafer-scale: 900K cores, 44GB SRAM, 4 Tflop.",
))
_reg(HardwareProfile(
    name="tesla_dojo3", vendor="Tesla", family="Dojo-3",
    platform_class="wafer_scale", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    dsp_block="TU", dsp_mult_a=16, dsp_mult_b=16,
    max_freq_mhz=2000,
    notes="Tesla Dojo 3: in-house wafer-scale AI supercomputer tile.",
))
_reg(HardwareProfile(
    name="tachyum_prodigy", vendor="Tachyum", family="Prodigy-2nm",
    platform_class="wafer_scale", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    max_freq_mhz=5500,
    notes="Tachyum Prodigy: universal processor. CPU+GPU+AI in one die.",
))

# ── Analog Mixed-Signal ──────────────────────────────────────────────

_reg(HardwareProfile(
    name="aspinity_aml100", vendor="Aspinity", family="AML100",
    platform_class="analog_mixed", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="Aspinity AML100: analog ML at the sensor. "
          "Always-on anomaly detection, µW power.",
))
_reg(HardwareProfile(
    name="renesas_analog_ai", vendor="Renesas", family="AnalogAI",
    platform_class="analog_mixed", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="Renesas Analog AI: mixed-signal compute-at-ADC boundary. "
          "Industrial IoT edge inference.",
))

# ── RRAM / Memristive Crossbar ───────────────────────────────────────

_reg(HardwareProfile(
    name="weebit_reram", vendor="Weebit Nano", family="ReRAM-ACiM",
    platform_class="rram", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="Weebit Nano ReRAM: analog compute-in-memory crossbar. "
          "200 TOPS/W, licensed to TI/Onsemi.",
))
_reg(HardwareProfile(
    name="crossbar_rram", vendor="Crossbar", family="ReRAM-1T1R",
    platform_class="rram", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="Crossbar Inc RRAM: 1T1R selector. Non-volatile weights, "
          "multi-level cell synapses.",
))
_reg(HardwareProfile(
    name="adesto_cbram", vendor="Adesto", family="CBRAM",
    platform_class="rram", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="Adesto CBRAM: conductive-bridge RAM. Low-power IoT NVM.",
))

# ── SRAM Compute-in-Memory ───────────────────────────────────────────

_reg(HardwareProfile(
    name="tsmc_cim_n7", vendor="TSMC", family="CIM-N7",
    platform_class="sram_cim", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    dsp_block="CIM_MACRO", dsp_mult_a=8, dsp_mult_b=8,
    max_freq_mhz=1000,
    notes="TSMC N7 SRAM-CIM macro: digital compute-in-memory. "
          "Standard cell integration, foundry-qualified.",
))
_reg(HardwareProfile(
    name="samsung_cim_sf3", vendor="Samsung", family="CIM-SF3",
    platform_class="sram_cim", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    dsp_block="CIM_MACRO", dsp_mult_a=8, dsp_mult_b=8,
    max_freq_mhz=900,
    notes="Samsung SF3 SRAM-CIM: 3nm GAA compute-in-memory macro.",
))

# ── Cryogenic CMOS ───────────────────────────────────────────────────

_reg(HardwareProfile(
    name="intel_horse_ridge", vendor="Intel", family="Horse-Ridge-II",
    platform_class="cryo_cmos", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    max_freq_mhz=6000,
    notes="Intel Horse Ridge II: cryogenic CMOS controller at 4K. "
          "Quantum-classical interface for neuromorphic control.",
))
_reg(HardwareProfile(
    name="google_cryo_ctrl", vendor="Google", family="Cryo-Controller",
    platform_class="cryo_cmos", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    max_freq_mhz=4000,
    notes="Google cryogenic controller: CMOS at 4K for quantum chip "
          "readout and control.",
))

# ── DNA / Molecular ──────────────────────────────────────────────────

_reg(HardwareProfile(
    name="microsoft_dna_store", vendor="Microsoft", family="DNA-Storage",
    platform_class="dna_molecular", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="Microsoft DNA Storage: enzymatic DNA synthesis for archival "
          "compute. Output = nucleotide sequence.",
))
_reg(HardwareProfile(
    name="asu_dna_perovskite", vendor="ASU", family="DNA-Perovskite",
    platform_class="dna_molecular", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="ASU DNA-Perovskite: bio-hybrid synaptic device. DNA base-pair "
          "gated perovskite semiconductor.",
))

# ── Quantum Neuromorphic ─────────────────────────────────────────────

_reg(HardwareProfile(
    name="ibm_qnn", vendor="IBM", family="Quantum-NN",
    platform_class="quantum_neuro", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    notes="IBM Quantum Neural Network: superconducting transmon qubits "
          "for quantum reservoir computing.",
))
_reg(HardwareProfile(
    name="ionq_trapped_ion", vendor="IonQ", family="Trapped-Ion-QNN",
    platform_class="quantum_neuro", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    notes="IonQ Trapped-Ion QNN: all-to-all connectivity for quantum "
          "SNN simulation.",
))

# ── Optical Interconnect / CPO ───────────────────────────────────────

_reg(HardwareProfile(
    name="ayar_teraphy", vendor="Ayar Labs", family="TeraPHY",
    platform_class="optical_io", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    max_freq_mhz=25000,
    notes="Ayar Labs TeraPHY: silicon photonic I/O chiplet. "
          "8 Tbps bidirectional, UCIe-compatible.",
))
_reg(HardwareProfile(
    name="intel_cpo", vendor="Intel", family="CPO",
    platform_class="optical_io", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    max_freq_mhz=20000,
    notes="Intel co-packaged optics: silicon photonic I/O for "
          "die-to-die and rack-scale optical links.",
))

# ── Acoustic / Phononic ──────────────────────────────────────────────

_reg(HardwareProfile(
    name="mit_phononic", vendor="MIT", family="Phononic-NN",
    platform_class="acoustic", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="MIT phononic neural network: acoustic wave reservoir "
          "computing in MEMS resonator arrays.",
))
_reg(HardwareProfile(
    name="caltech_mems_nn", vendor="Caltech", family="MEMS-NN",
    platform_class="acoustic", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="Caltech MEMS neural processor: mechanical resonator "
          "array for edge inference.",
))

# ── Fluidic / Microfluidic ───────────────────────────────────────────

_reg(HardwareProfile(
    name="stanford_microfluidic", vendor="Stanford", family="µFluidic-NN",
    platform_class="fluidic", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="Stanford microfluidic neural network: droplet-based "
          "logic gates for lab-on-chip compute.",
))
_reg(HardwareProfile(
    name="eth_fluidic_logic", vendor="ETH Zurich", family="Fluidic-Logic",
    platform_class="fluidic", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="ETH Zurich fluidic logic: pressure-driven bistable "
          "valves for chemical neural computation.",
))

# ── Space-Qualified ──────────────────────────────────────────────────

_reg(HardwareProfile(
    name="bae_rad750_sq", vendor="BAE Systems", family="RAD750",
    platform_class="space_qualified", data_width=32, fraction=16,
    overflow="saturate", rounding="nearest",
    max_freq_mhz=200,
    notes="BAE RAD750: radiation-hardened processor. Mars rovers, "
          "ISS, deep-space missions.",
))
_reg(HardwareProfile(
    name="seakr_sbc", vendor="SEAKR", family="SBC-SpaceAI",
    platform_class="space_qualified", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    max_freq_mhz=400,
    notes="SEAKR SpaceAI SBC: radiation-tolerant single-board "
          "computer for on-orbit neural inference.",
))
_reg(HardwareProfile(
    name="vorago_va10820", vendor="Vorago", family="VA10820",
    platform_class="space_qualified", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    max_freq_mhz=100,
    notes="Vorago VA10820: Arm Cortex-M0 rad-hard MCU for "
          "space-grade edge neural processing.",
))
_reg(HardwareProfile(
    name="frontgrade_leon5", vendor="Frontgrade", family="LEON5-FT",
    platform_class="space_qualified", data_width=32, fraction=16,
    overflow="saturate", rounding="nearest",
    max_freq_mhz=250,
    notes="Frontgrade LEON5-FT: SPARC V8 rad-hard for ESA/NASA "
          "mission-critical neural control.",
))

# ── Wave 10: Magnonic / Skyrmion ─────────────────────────────────────

_reg(HardwareProfile(
    name="tum_skyrmion", vendor="TU Munich", family="SkyANN-v1",
    platform_class="magnonic", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="Skyrmion-based reservoir computing. EU SkyANN project. "
          "Topological stability enables ultra-low-power edge AI.",
))

_reg(HardwareProfile(
    name="kaist_spinwave", vendor="KAIST", family="SpinWave-RC",
    platform_class="magnonic", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="Spin-wave interference reservoir. Field-free operation via "
          "SOT bilayer nanostructures.",
))

_reg(HardwareProfile(
    name="imec_mtj_reservoir", vendor="imec", family="MTJ-Reservoir",
    platform_class="magnonic", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="Magnetic tunnel junction reservoir computing array. "
          "Sub-fJ switching energy per MAC operation.",
))

# ── Wave 10: Organic Bioelectronic ───────────────────────────────────

_reg(HardwareProfile(
    name="cambridge_oect", vendor="Cambridge", family="OECT-Synapse",
    platform_class="organic_bioelectronic", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="Organic Electrochemical Transistor synapse. PEDOT:PSS "
          "channel for in-vivo bioelectronic neural interfaces.",
))

_reg(HardwareProfile(
    name="linkoping_organic", vendor="Linköping", family="Organic-NN",
    platform_class="organic_bioelectronic", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="Printed organic transistor array. Biodegradable substrate "
          "for disposable sensor-neural-interface.",
))

# ── Wave 10: RISC-V Sovereign AI ─────────────────────────────────────

_reg(HardwareProfile(
    name="sifive_x280_ai", vendor="SiFive", family="X280-AI",
    platform_class="risc_v_sovereign", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    max_freq_mhz=2000,
    notes="SiFive Intelligence X280: RISC-V vector AI core. "
          "Open ISA, no ITAR restrictions, sovereign compute.",
))

_reg(HardwareProfile(
    name="esperanto_et_soc", vendor="Esperanto", family="ET-SoC-1",
    platform_class="risc_v_sovereign", data_width=8, fraction=4,
    overflow="saturate", rounding="nearest",
    max_freq_mhz=1000,
    notes="Esperanto ET-SoC-1: 1000+ RISC-V cores for sovereign "
          "AI inference. No export control dependencies.",
))

_reg(HardwareProfile(
    name="ventana_veyron_ai", vendor="Ventana", family="Veyron-V2",
    platform_class="risc_v_sovereign", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    max_freq_mhz=3600,
    notes="Ventana Veyron V2: high-perf RISC-V with AI extensions. "
          "Chiplet-based, UCIe-compatible.",
))

_reg(HardwareProfile(
    name="tenstorrent_ascalon", vendor="Tenstorrent", family="Ascalon",
    platform_class="risc_v_sovereign", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    max_freq_mhz=4000,
    notes="Tenstorrent Ascalon: RISC-V AI server processor. "
          "Open-source ISA for data-sovereign deployments.",
))

_reg(HardwareProfile(
    name="andes_ax45mpv", vendor="Andes", family="AX45MPV",
    platform_class="risc_v_sovereign", data_width=16, fraction=8,
    overflow="saturate", rounding="nearest",
    max_freq_mhz=1500,
    notes="Andes AX45MPV: multiprocessor RISC-V with vector extension. "
          "Targets automotive and edge AI sovereignty.",
))

# ── Wave 11: Thermodynamic Computing ─────────────────────────────────

_reg(HardwareProfile(
    name="extropic_epu", vendor="Extropic", family="EPU-v1",
    platform_class="thermodynamic", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="Extropic Energy-Based Processor: probabilistic generative AI "
          "via controlled thermal fluctuations. Room-temperature.",
))

_reg(HardwareProfile(
    name="normal_cn101", vendor="Normal Computing", family="CN101",
    platform_class="thermodynamic", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="Normal Computing CN101: thermodynamic AI chip. Stochastic "
          "sampling via thermal noise exploitation.",
))

# ── Wave 11: Probabilistic / p-Bit ───────────────────────────────────

_reg(HardwareProfile(
    name="purdue_pbit", vendor="Purdue", family="p-Bit-Array",
    platform_class="probabilistic", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="Purdue MRAM p-bit array: room-temperature probabilistic "
          "computing. Boltzmann machine substrate.",
))

_reg(HardwareProfile(
    name="tohoku_sot_pbit", vendor="Tohoku", family="SOT-pBit",
    platform_class="probabilistic", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="Tohoku SOT-MRAM probabilistic computing: tuneable "
          "fluctuation rate via spin-orbit torque bias.",
))

# ── Wave 11: Polariton / Exciton ─────────────────────────────────────

_reg(HardwareProfile(
    name="marvell_polariton", vendor="Marvell", family="Polariton-PIC",
    platform_class="polariton", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="Marvell/Polariton Technologies: silicon photonic + plasmonic "
          "active devices for ultrafast optical neural compute.",
))

_reg(HardwareProfile(
    name="stanford_polariton", vendor="Stanford", family="Perovskite-RC",
    platform_class="polariton", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="Stanford perovskite microcavity: exciton-polariton "
          "condensate reservoir computing at room temperature.",
))

# ── Wave 11: Metamaterial / Programmable Matter ──────────────────────

_reg(HardwareProfile(
    name="mit_metamaterial", vendor="MIT", family="RF-Metasurface",
    platform_class="metamaterial", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="MIT RF metasurface neural network: programmable "
          "unit-cell phases perform analog matrix-vector multiply.",
))

_reg(HardwareProfile(
    name="penn_acoustic_meta", vendor="UPenn", family="Acoustic-Meta",
    platform_class="metamaterial", data_width=8, fraction=4,
    overflow="saturate", rounding="truncate",
    notes="UPenn acoustic metamaterial classifier: mechanical "
          "wave propagation implements inference at zero digital power.",
))

# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════

def get_profile(name: str) -> HardwareProfile:
    """Look up a hardware profile by name.

    Parameters
    ----------
    name : str
        Case-insensitive profile name (e.g. ``"loihi2"``, ``"artix7"``).

    Returns
    -------
    HardwareProfile
        The matching profile.

    Raises
    ------
    KeyError
        If no profile matches.
    """
    key = name.lower().replace("-", "_").replace(" ", "_")
    if key not in _PROFILES:
        available = ", ".join(sorted(_PROFILES.keys()))
        raise KeyError(
            f"Unknown hardware profile '{name}'. "
            f"Available: {available}"
        )
    return _PROFILES[key]


def list_profiles(
    *,
    platform_class: str | None = None,
    vendor: str | None = None,
) -> list[HardwareProfile]:
    """List all registered hardware profiles, optionally filtered.

    Parameters
    ----------
    platform_class : str, optional
        Filter by class: ``"fpga"``, ``"neuromorphic"``, ``"asic"``, ``"simulation"``
        ``"accelerator"``, ``"dsp"``, ``"photonic"``, ``"in_memory"``, ``"emerging"``.
    vendor : str, optional
        Filter by vendor name (case-insensitive substring match).

    Returns
    -------
    list[HardwareProfile]
        Matching profiles, sorted by (platform_class, vendor, name).
    """
    result = list(_PROFILES.values())
    if platform_class:
        result = [p for p in result if p.platform_class == platform_class]
    if vendor:
        v_lower = vendor.lower()
        result = [p for p in result if v_lower in p.vendor.lower()]
    return sorted(result, key=lambda p: (p.platform_class, p.vendor, p.name))


def list_profile_names() -> list[str]:
    """Return all registered profile names, sorted."""
    return sorted(_PROFILES.keys())


def load_toml_profile(path: str) -> HardwareProfile:
    """Load a user-defined hardware profile from a TOML file.

    Enables users to register custom hardware targets without modifying
    the SC-NeuroCore source. TOML format::

        [profile]
        name = "my_chip"
        vendor = "My Corp"
        family = "ChipNet-1"
        platform_class = "accelerator"
        data_width = 16
        fraction = 8
        overflow = "saturate"
        rounding = "nearest"
        max_freq_mhz = 500
        dsp_block = "MAC"
        dsp_mult_a = 16
        dsp_mult_b = 16
        notes = "Custom chip description."

    Parameters
    ----------
    path : str
        Path to the TOML file.

    Returns
    -------
    HardwareProfile
        The loaded and registered profile.

    Raises
    ------
    FileNotFoundError
        If the TOML file does not exist.
    ValueError
        If required fields are missing.
    """
    import tomllib
    from pathlib import Path

    toml_path = Path(path)
    if not toml_path.exists():
        raise FileNotFoundError(f"Profile TOML not found: {path}")

    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    p = data.get("profile", data)

    required = {"name", "vendor", "family", "platform_class",
                "data_width", "fraction", "overflow", "rounding"}
    missing = required - set(p.keys())
    if missing:
        raise ValueError(f"Missing required fields in TOML profile: {missing}")

    profile = HardwareProfile(
        name=p["name"],
        vendor=p["vendor"],
        family=p["family"],
        platform_class=p["platform_class"],
        data_width=int(p["data_width"]),
        fraction=int(p["fraction"]),
        overflow=p["overflow"],
        rounding=p["rounding"],
        max_freq_mhz=int(p.get("max_freq_mhz", 0)) or None,
        dsp_block=p.get("dsp_block", ""),
        dsp_mult_a=int(p.get("dsp_mult_a", 0)),
        dsp_mult_b=int(p.get("dsp_mult_b", 0)),
        notes=p.get("notes", "User-defined profile."),
    )
    _reg(profile)
    return profile


def load_toml_profiles_dir(directory: str) -> list[HardwareProfile]:
    """Load all TOML profiles from a directory.

    Scans the directory for ``*.toml`` files and loads each as a hardware
    profile. Useful for bulk-registering custom targets.

    Parameters
    ----------
    directory : str
        Path to the directory containing TOML profile files.

    Returns
    -------
    list[HardwareProfile]
        All loaded profiles.
    """
    from pathlib import Path

    profiles = []
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return profiles
    for toml_file in sorted(dir_path.glob("*.toml")):
        profiles.append(load_toml_profile(str(toml_file)))
    return profiles
