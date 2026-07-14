# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — programmable and aerospace profile registrations

"""Register programmable-logic and aerospace hardware profiles."""

from __future__ import annotations

from .registry import HardwareProfile, _reg


def _register_additional_fpga_profiles() -> None:
    """Register additional programmable-logic family profiles."""
    _reg(
        HardwareProfile(
            name="versal_ai_edge",
            vendor="AMD/Xilinx",
            family="Versal AI Edge",
            platform_class="fpga",
            data_width=16,
            fraction=8,
            dsp_block="DSP58",
            dsp_mult_a=27,
            dsp_mult_b=24,
            max_freq_mhz=900,
            overflow="saturate",
            rounding="nearest",
            notes="Versal AI Edge: AI Engines + PL. DSP58 = 27x24.",
        )
    )
    _reg(
        HardwareProfile(
            name="proasic3",
            vendor="Microchip",
            family="ProASIC3",
            platform_class="fpga",
            data_width=16,
            fraction=8,
            overflow="saturate",
            rounding="truncate",
            notes="Flash-based, no bitstream volatility. Aerospace legacy.",
        )
    )
    _reg(
        HardwareProfile(
            name="trion",
            vendor="Efinix",
            family="Trion T-series",
            platform_class="fpga",
            data_width=16,
            fraction=8,
            dsp_block="MULT",
            dsp_mult_a=10,
            dsp_mult_b=10,
            max_freq_mhz=200,
            overflow="saturate",
            rounding="truncate",
            notes="Efinix Trion: ultra-low-cost Quantum fabric. 10x10 mult.",
        )
    )
    _reg(
        HardwareProfile(
            name="titanium",
            vendor="Efinix",
            family="Titanium Ti-series",
            platform_class="fpga",
            data_width=18,
            fraction=9,
            dsp_block="MULT",
            dsp_mult_a=18,
            dsp_mult_b=18,
            max_freq_mhz=500,
            overflow="saturate",
            rounding="nearest",
            notes="Efinix Titanium: hardened RISC-V core + 18x18 mult.",
        )
    )
    _reg(
        HardwareProfile(
            name="gowin_arora_v",
            vendor="Gowin",
            family="Arora V GW5A",
            platform_class="fpga",
            data_width=18,
            fraction=9,
            dsp_block="DSP",
            dsp_mult_a=18,
            dsp_mult_b=18,
            max_freq_mhz=400,
            overflow="saturate",
            rounding="truncate",
            notes="Gowin Arora V 28nm: premium tier with 18x18 DSP.",
        )
    )
    _reg(
        HardwareProfile(
            name="intel_agilex5",
            vendor="Intel",
            family="Agilex 5 E/D",
            platform_class="fpga",
            data_width=18,
            fraction=9,
            dsp_block="DSP",
            dsp_mult_a=18,
            dsp_mult_b=19,
            max_freq_mhz=800,
            overflow="saturate",
            rounding="nearest",
            notes="Intel Agilex 5 E/D-series, HBM2e. Latest mid-range.",
        )
    )


def _register_aerospace_profiles() -> None:
    """Register sovereign, defence, and aerospace hardware profiles."""
    _reg(
        HardwareProfile(
            name="bae_rad750",
            vendor="BAE Systems",
            family="RAD750",
            platform_class="fpga",
            data_width=32,
            fraction=16,
            overflow="saturate",
            rounding="truncate",
            dsp_block="",
            dsp_mult_a=0,
            dsp_mult_b=0,
            notes="BAE RAD750: rad-hard PowerPC. Mars rovers, deep space.",
        )
    )
    _reg(
        HardwareProfile(
            name="cobham_ut700",
            vendor="Cobham/CAES",
            family="UT700 LEON3",
            platform_class="fpga",
            data_width=32,
            fraction=16,
            overflow="saturate",
            rounding="truncate",
            dsp_block="",
            dsp_mult_a=0,
            dsp_mult_b=0,
            notes="Cobham UT700: LEON3 SPARC rad-hard. ESA space missions.",
        )
    )
    _reg(
        HardwareProfile(
            name="mpfs250t_rt",
            vendor="Microchip",
            family="PolarFire SoC RT",
            platform_class="fpga",
            data_width=24,
            fraction=12,
            overflow="saturate",
            rounding="nearest",
            dsp_block="MACC",
            dsp_mult_a=18,
            dsp_mult_b=18,
            max_freq_mhz=250,
            notes="PolarFire SoC RT: rad-tolerant RISC-V + FPGA. LEO satellites.",
        )
    )
    _reg(
        HardwareProfile(
            name="versal_xqrvc1902",
            vendor="AMD/Xilinx",
            family="Versal RT",
            platform_class="fpga",
            data_width=32,
            fraction=16,
            overflow="saturate",
            rounding="nearest",
            dsp_block="DSP58",
            dsp_mult_a=27,
            dsp_mult_b=24,
            max_freq_mhz=600,
            notes="AMD Versal XQRVC1902: rad-hard Versal AI Core for satellites.",
        )
    )
    _reg(
        HardwareProfile(
            name="trenz_zynq_space",
            vendor="Trenz",
            family="Zynq Space-Grade",
            platform_class="fpga",
            data_width=16,
            fraction=8,
            overflow="saturate",
            rounding="nearest",
            dsp_block="DSP48E1",
            dsp_mult_a=25,
            dsp_mult_b=18,
            max_freq_mhz=300,
            notes="Trenz space-grade Zynq module: ESA-qualified for LEO/GEO.",
        )
    )
