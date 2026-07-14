# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FPGA hardware profile registrations

"""Register discrete, radiation-hardened, and embedded FPGA profiles."""

from __future__ import annotations

from .registry import HardwareProfile, _reg

# ── Xilinx FPGA ─────────────────────────────────────────────────────


def _register_xilinx_fpga_profiles() -> None:
    """Register Xilinx FPGA profiles."""
    _reg(
        HardwareProfile(
            name="spartan6",
            vendor="Xilinx",
            family="Spartan-6",
            platform_class="fpga",
            data_width=18,
            fraction=9,
            dsp_block="DSP48A1",
            dsp_mult_a=18,
            dsp_mult_b=18,
            max_freq_mhz=250,
            notes="18-bit DSP48A1 native. Q9.9 uses full B-port.",
        )
    )

    _reg(
        HardwareProfile(
            name="artix7",
            vendor="Xilinx",
            family="Artix-7",
            platform_class="fpga",
            data_width=18,
            fraction=9,
            dsp_block="DSP48E1",
            dsp_mult_a=25,
            dsp_mult_b=18,
            max_freq_mhz=450,
            notes="Q9.9 uses full 18-bit B-port. A-port has 7 guard bits.",
        )
    )

    _reg(
        HardwareProfile(
            name="kintex7",
            vendor="Xilinx",
            family="Kintex-7",
            platform_class="fpga",
            data_width=18,
            fraction=9,
            dsp_block="DSP48E1",
            dsp_mult_a=25,
            dsp_mult_b=18,
            max_freq_mhz=500,
            notes="Same DSP48E1 as Artix-7 but faster fabric.",
        )
    )

    _reg(
        HardwareProfile(
            name="ultrascale",
            vendor="Xilinx",
            family="UltraScale",
            platform_class="fpga",
            data_width=36,
            fraction=18,
            dsp_block="DSP48E2",
            dsp_mult_a=27,
            dsp_mult_b=18,
            max_freq_mhz=600,
            notes="Q18.18 uses full 18-bit B-port × 2 for 36-bit state.",
        )
    )

    _reg(
        HardwareProfile(
            name="ultrascale_plus",
            vendor="Xilinx",
            family="UltraScale+",
            platform_class="fpga",
            data_width=36,
            fraction=18,
            dsp_block="DSP48E2",
            dsp_mult_a=27,
            dsp_mult_b=18,
            max_freq_mhz=775,
            notes="Same DSP48E2 architecture, higher speed grade.",
        )
    )

    _reg(
        HardwareProfile(
            name="versal",
            vendor="Xilinx",
            family="Versal",
            platform_class="fpga",
            data_width=24,
            fraction=12,
            dsp_block="DSP58",
            dsp_mult_a=27,
            dsp_mult_b=24,
            max_freq_mhz=900,
            notes="DSP58 extends B-port to 24 bits. Q12.12 is native.",
        )
    )


# ── Intel FPGA ───────────────────────────────────────────────────────


def _register_intel_fpga_profiles() -> None:
    """Register Intel FPGA profiles."""
    _reg(
        HardwareProfile(
            name="cyclone_v",
            vendor="Intel",
            family="Cyclone V",
            platform_class="fpga",
            data_width=18,
            fraction=9,
            dsp_block="Variable Precision",
            dsp_mult_a=18,
            dsp_mult_b=18,
            max_freq_mhz=300,
            notes="18×18 native. Same Q9.9 as Xilinx 7-series.",
        )
    )

    _reg(
        HardwareProfile(
            name="cyclone_10",
            vendor="Intel",
            family="Cyclone 10",
            platform_class="fpga",
            data_width=18,
            fraction=9,
            dsp_block="Variable Precision",
            dsp_mult_a=18,
            dsp_mult_b=18,
            max_freq_mhz=300,
            notes="Same DSP as Cyclone V.",
        )
    )

    _reg(
        HardwareProfile(
            name="arria10",
            vendor="Intel",
            family="Arria 10",
            platform_class="fpga",
            data_width=27,
            fraction=13,
            dsp_block="Variable Precision",
            dsp_mult_a=27,
            dsp_mult_b=27,
            max_freq_mhz=500,
            notes="27×27 native. Q14.13 uses full multiplier width.",
        )
    )

    _reg(
        HardwareProfile(
            name="stratix10",
            vendor="Intel",
            family="Stratix 10",
            platform_class="fpga",
            data_width=27,
            fraction=13,
            dsp_block="Variable Precision",
            dsp_mult_a=27,
            dsp_mult_b=27,
            max_freq_mhz=700,
            notes="Same 27×27 DSP as Arria 10, higher clock.",
        )
    )

    _reg(
        HardwareProfile(
            name="agilex",
            vendor="Intel",
            family="Agilex",
            platform_class="fpga",
            data_width=27,
            fraction=13,
            dsp_block="Variable Precision",
            dsp_mult_a=27,
            dsp_mult_b=27,
            max_freq_mhz=800,
            notes="Latest Intel 10nm FPGA. 27×27 native DSP.",
        )
    )


# ── Lattice FPGA ─────────────────────────────────────────────────────


def _register_lattice_fpga_profiles() -> None:
    """Register Lattice FPGA profiles."""
    _reg(
        HardwareProfile(
            name="ecp5",
            vendor="Lattice",
            family="ECP5",
            platform_class="fpga",
            data_width=18,
            fraction=9,
            dsp_block="MULT18X18D",
            dsp_mult_a=18,
            dsp_mult_b=18,
            max_freq_mhz=400,
            notes="18×18 native. Popular open-source FPGA (Yosys/nextpnr).",
        )
    )

    _reg(
        HardwareProfile(
            name="crosslink_nx",
            vendor="Lattice",
            family="CrossLink-NX",
            platform_class="fpga",
            data_width=18,
            fraction=9,
            dsp_block="DSP",
            dsp_mult_a=18,
            dsp_mult_b=18,
            max_freq_mhz=400,
            notes="28nm, low power. Same 18×18 multiplier as ECP5.",
        )
    )

    _reg(
        HardwareProfile(
            name="certuspro_nx",
            vendor="Lattice",
            family="CertusPro-NX",
            platform_class="fpga",
            data_width=18,
            fraction=9,
            dsp_block="DSP",
            dsp_mult_a=18,
            dsp_mult_b=18,
            max_freq_mhz=500,
            notes="Lattice's largest 28nm FPGA. 18×18 DSP.",
        )
    )


# ── Other FPGA Vendors ──────────────────────────────────────────────


def _register_other_fpga_profiles() -> None:
    """Register FPGA profiles from other vendors."""
    _reg(
        HardwareProfile(
            name="gowin",
            vendor="Gowin",
            family="GW1N/GW2A",
            platform_class="fpga",
            data_width=18,
            fraction=9,
            dsp_block="MULT18X18",
            dsp_mult_a=18,
            dsp_mult_b=18,
            max_freq_mhz=300,
            notes="Chinese FPGA. 18×18 native, Yosys-compatible.",
        )
    )

    _reg(
        HardwareProfile(
            name="efinix",
            vendor="Efinix",
            family="Trion/Titanium",
            platform_class="fpga",
            data_width=10,
            fraction=5,
            dsp_block="MULT",
            dsp_mult_a=10,
            dsp_mult_b=9,
            max_freq_mhz=300,
            notes="10×9 multiplier. Q5.5 (10-bit) is the native maximum.",
        )
    )

    _reg(
        HardwareProfile(
            name="polarfire",
            vendor="Microchip",
            family="PolarFire",
            platform_class="fpga",
            data_width=18,
            fraction=9,
            dsp_block="MACC",
            dsp_mult_a=18,
            dsp_mult_b=18,
            max_freq_mhz=400,
            notes="Radiation-tolerant. 18×18 MACC block.",
        )
    )

    _reg(
        HardwareProfile(
            name="smartfusion2",
            vendor="Microchip",
            family="SmartFusion2",
            platform_class="fpga",
            data_width=18,
            fraction=9,
            dsp_block="MACC",
            dsp_mult_a=18,
            dsp_mult_b=18,
            max_freq_mhz=200,
            notes="FPGA + ARM Cortex-M3. 18×18 MACC.",
        )
    )

    _reg(
        HardwareProfile(
            name="achronix",
            vendor="Achronix",
            family="Speedster7t",
            platform_class="fpga",
            data_width=24,
            fraction=12,
            dsp_block="MLP",
            dsp_mult_a=24,
            dsp_mult_b=24,
            max_freq_mhz=750,
            notes="Machine Learning Processor block. 24×24 native.",
        )
    )

    _reg(
        HardwareProfile(
            name="quicklogic",
            vendor="QuickLogic",
            family="EOS S3",
            platform_class="fpga",
            data_width=8,
            fraction=4,
            dsp_block="LUT-based",
            dsp_mult_a=0,
            dsp_mult_b=0,
            max_freq_mhz=80,
            notes="No DSP blocks. LUT-based multiply. 8-bit optimal.",
        )
    )


# ── ICE40 (Lattice, open-source) ────────────────────────────────────


def _register_ice40_fpga_profiles() -> None:
    """Register the Lattice iCE40 FPGA profile."""
    _reg(
        HardwareProfile(
            name="ice40",
            vendor="Lattice",
            family="iCE40",
            platform_class="fpga",
            data_width=16,
            fraction=8,
            dsp_block="SB_MAC16",
            dsp_mult_a=16,
            dsp_mult_b=16,
            max_freq_mhz=270,
            notes="Tiny open-source FPGA. 16×16 SB_MAC16. Q8.8 native.",
        )
    )


# ── Additional FPGA (2025–2026 platforms) ────────────────────────────


def _register_additional_fpga_profiles() -> None:
    """Register additional current-generation FPGA profiles."""
    _reg(
        HardwareProfile(
            name="alveo",
            vendor="AMD/Xilinx",
            family="Alveo U50/U200/U280",
            platform_class="fpga",
            data_width=36,
            fraction=18,
            dsp_block="DSP48E2",
            dsp_mult_a=27,
            dsp_mult_b=18,
            max_freq_mhz=700,
            notes="Data-centre accelerator card. Same DSP48E2 as UltraScale+.",
        )
    )

    _reg(
        HardwareProfile(
            name="nexus",
            vendor="Lattice",
            family="Nexus (LIFCL)",
            platform_class="fpga",
            data_width=18,
            fraction=9,
            dsp_block="DSP",
            dsp_mult_a=18,
            dsp_mult_b=18,
            max_freq_mhz=500,
            notes="28nm replacement for ECP5. 18×18 DSP.",
        )
    )

    _reg(
        HardwareProfile(
            name="polarfire_soc",
            vendor="Microchip",
            family="PolarFire SoC",
            platform_class="fpga",
            data_width=18,
            fraction=9,
            dsp_block="MACC",
            dsp_mult_a=18,
            dsp_mult_b=18,
            max_freq_mhz=400,
            notes="FPGA + RISC-V hard core. 18×18 MACC. ARM-free.",
        )
    )

    _reg(
        HardwareProfile(
            name="avant",
            vendor="Lattice",
            family="Avant",
            platform_class="fpga",
            data_width=18,
            fraction=9,
            dsp_block="DSP",
            dsp_mult_a=18,
            dsp_mult_b=18,
            max_freq_mhz=500,
            notes="Next-gen Lattice 16nm mid-range FPGA.",
        )
    )


# ── Radiation-Hardened / Space ───────────────────────────────────────


def _register_radiation_hardened_fpga_profiles() -> None:
    """Register radiation-hardened FPGA profiles."""
    _reg(
        HardwareProfile(
            name="nanoxplore",
            vendor="NanoXplore",
            family="NG-Ultra",
            platform_class="fpga",
            data_width=18,
            fraction=9,
            dsp_block="DSP",
            dsp_mult_a=18,
            dsp_mult_b=18,
            overflow="trap",
            rounding="nearest",
            max_freq_mhz=300,
            notes="European rad-hard FPGA (ESA approved). SEU-tolerant. 18×18 DSP.",
        )
    )

    _reg(
        HardwareProfile(
            name="rtg4",
            vendor="Microchip",
            family="RTG4",
            platform_class="fpga",
            data_width=18,
            fraction=9,
            dsp_block="MACC",
            dsp_mult_a=18,
            dsp_mult_b=18,
            overflow="trap",
            rounding="nearest",
            max_freq_mhz=200,
            notes="Rad-tolerant flash FPGA. NASA/ESA heritage. 18×18 MACC.",
        )
    )

    _reg(
        HardwareProfile(
            name="kintex_us_rt",
            vendor="Xilinx",
            family="Kintex UltraScale RT",
            platform_class="fpga",
            data_width=36,
            fraction=18,
            dsp_block="DSP48E2",
            dsp_mult_a=27,
            dsp_mult_b=18,
            overflow="trap",
            rounding="nearest",
            max_freq_mhz=500,
            notes="Space-grade Kintex (XQRKU060). TMR-compatible. 27×18 DSP.",
        )
    )


# ── Embedded FPGA (eFPGA) IP ────────────────────────────────────────


def _register_embedded_fpga_profiles() -> None:
    """Register embedded-FPGA IP profiles."""
    _reg(
        HardwareProfile(
            name="speedcore",
            vendor="Achronix",
            family="Speedcore eFPGA IP",
            platform_class="fpga",
            data_width=24,
            fraction=12,
            dsp_block="MLP",
            dsp_mult_a=24,
            dsp_mult_b=24,
            max_freq_mhz=750,
            notes="eFPGA IP embedded in custom ASICs. 24×24 MLP blocks.",
        )
    )

    _reg(
        HardwareProfile(
            name="eflx",
            vendor="Flex Logix",
            family="EFLX eFPGA",
            platform_class="fpga",
            data_width=16,
            fraction=8,
            dsp_block="LUT-based",
            dsp_mult_a=0,
            dsp_mult_b=0,
            max_freq_mhz=500,
            notes="eFPGA for post-silicon reconfigurability. LUT-based multiply.",
        )
    )

    _reg(
        HardwareProfile(
            name="menta_efpga",
            vendor="Menta",
            family="Origami eFPGA",
            platform_class="fpga",
            data_width=16,
            fraction=8,
            dsp_block="LUT-based",
            dsp_mult_a=0,
            dsp_mult_b=0,
            max_freq_mhz=400,
            notes="European eFPGA IP. LUT-based arithmetic.",
        )
    )
