# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore

"""Register CMOS, FPGA, ASIC, MCU, and conventional accelerator profiles."""

from __future__ import annotations

from .registry import HardwareProfile, _reg

# ── Xilinx FPGA ─────────────────────────────────────────────────────

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

# ── ASIC Targets ─────────────────────────────────────────────────────

_reg(
    HardwareProfile(
        name="asic_16",
        vendor="ASIC",
        family="Standard Cell (16-bit)",
        platform_class="asic",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        notes="Generic 16-bit ASIC. No DSP constraint — any width is synthesisable.",
    )
)

_reg(
    HardwareProfile(
        name="asic_32",
        vendor="ASIC",
        family="Standard Cell (32-bit)",
        platform_class="asic",
        data_width=32,
        fraction=16,
        overflow="saturate",
        rounding="nearest",
        notes="Generic 32-bit ASIC. Q16.16 gold standard.",
    )
)

_reg(
    HardwareProfile(
        name="asic_custom",
        vendor="ASIC",
        family="Custom",
        platform_class="asic",
        data_width=24,
        fraction=12,
        overflow="trap",
        rounding="bankers",
        notes="Safety-critical ASIC (DO-254/IEC 61508). Trap on overflow.",
    )
)

# ── Simulation / Golden Reference ────────────────────────────────────

_reg(
    HardwareProfile(
        name="sim_q88",
        vendor="Simulation",
        family="Icarus Q8.8",
        platform_class="simulation",
        data_width=16,
        fraction=8,
        notes="Default simulation target for iverilog co-simulation.",
    )
)

_reg(
    HardwareProfile(
        name="sim_q1616",
        vendor="Simulation",
        family="Icarus Q16.16",
        platform_class="simulation",
        data_width=32,
        fraction=16,
        notes="Gold standard simulation for fidelity validation.",
    )
)

# ── Additional FPGA (2025–2026 platforms) ────────────────────────────

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

# ── AI / ML Accelerators ────────────────────────────────────────────

_reg(
    HardwareProfile(
        name="tpu",
        vendor="Google",
        family="TPU v4/v5",
        platform_class="accelerator",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        notes="Tensor Processing Unit. INT8/bfloat16. Q8.8 for param transfer.",
    )
)

_reg(
    HardwareProfile(
        name="cerebras_wse",
        vendor="Cerebras",
        family="WSE-3",
        platform_class="accelerator",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        notes="Wafer-scale engine. 900K cores, FP16 native.",
    )
)

_reg(
    HardwareProfile(
        name="graphcore_ipu",
        vendor="Graphcore",
        family="IPU Mk2/Bow",
        platform_class="accelerator",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        notes="Intelligence Processing Unit. FP16 accumulation.",
    )
)

_reg(
    HardwareProfile(
        name="tenstorrent",
        vendor="Tenstorrent",
        family="Grayskull/Wormhole",
        platform_class="accelerator",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="truncate",
        notes="RISC-V based AI accelerator. 8/16-bit tensor ops.",
    )
)

_reg(
    HardwareProfile(
        name="ethos_u",
        vendor="ARM",
        family="Ethos-U55/U65",
        platform_class="accelerator",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="nearest",
        notes="Micro NPU for Cortex-M edge AI. 8-bit native.",
    )
)

_reg(
    HardwareProfile(
        name="hexagon",
        vendor="Qualcomm",
        family="Hexagon DSP/HVX",
        platform_class="accelerator",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        notes="Mobile DSP with 16-bit HVX SIMD.",
    )
)

_reg(
    HardwareProfile(
        name="apple_ane",
        vendor="Apple",
        family="Neural Engine (M-series)",
        platform_class="accelerator",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        notes="Apple Neural Engine. 16-bit inference.",
    )
)

# ── DSP Processors ──────────────────────────────────────────────────

_reg(
    HardwareProfile(
        name="sharc",
        vendor="Analog Devices",
        family="SHARC ADSP-SC5xx",
        platform_class="dsp",
        data_width=32,
        fraction=16,
        overflow="saturate",
        rounding="nearest",
        notes="32/40-bit fixed-point DSP. Q16.16 native.",
    )
)

_reg(
    HardwareProfile(
        name="c6000",
        vendor="Texas Instruments",
        family="C66x/C67x",
        platform_class="dsp",
        data_width=32,
        fraction=16,
        overflow="saturate",
        rounding="nearest",
        notes="40-bit accumulator. Q16.16 for fixed-point mode.",
    )
)

_reg(
    HardwareProfile(
        name="ceva_xc",
        vendor="CEVA",
        family="CEVA-XC/X2",
        platform_class="dsp",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        notes="Communications DSP. 16-bit Q8.8 for real-time.",
    )
)

# ── Emerging Compute Paradigms ───────────────────────────────────────

_reg(
    HardwareProfile(
        name="photonic",
        vendor="Lightmatter",
        family="Envise/Passage",
        platform_class="emerging",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="Optical matrix multiply. 8-bit DAC/ADC resolution.",
    )
)

_reg(
    HardwareProfile(
        name="riscv_fpga",
        vendor="SiFive",
        family="RISC-V + FPGA fabric",
        platform_class="emerging",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        notes="Open ISA with custom neural extensions. Q8.8 baseline.",
    )
)

_reg(
    HardwareProfile(
        name="in_memory",
        vendor="Mythic/Syntiant",
        family="Analog In-Memory",
        platform_class="emerging",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="Computation in flash/SRAM. 4-8 bit effective resolution.",
    )
)

_reg(
    HardwareProfile(
        name="quantum_hybrid",
        vendor="IBM/Custom",
        family="Quantum-Classical",
        platform_class="emerging",
        data_width=32,
        fraction=16,
        overflow="saturate",
        rounding="bankers",
        notes="Quantum-classical interface. Q16.16 for parameter encoding.",
    )
)

# ── Radiation-Hardened / Space ───────────────────────────────────────

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

# ── Edge AI Accelerators ────────────────────────────────────────────

_reg(
    HardwareProfile(
        name="hailo8",
        vendor="Hailo",
        family="Hailo-8/8L",
        platform_class="accelerator",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="nearest",
        notes="26 TOPS edge AI. Dominant in automotive ADAS.",
    )
)

_reg(
    HardwareProfile(
        name="kneron",
        vendor="Kneron",
        family="KL730",
        platform_class="accelerator",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="nearest",
        notes="Ultra-low-power edge NPU. 4-bit/8-bit modes.",
    )
)

_reg(
    HardwareProfile(
        name="groq_tsp",
        vendor="Groq",
        family="TSP (LPU)",
        platform_class="accelerator",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        notes="Tensor Streaming Processor. Deterministic latency, no caches.",
    )
)

_reg(
    HardwareProfile(
        name="jetson",
        vendor="NVIDIA",
        family="Jetson Orin",
        platform_class="accelerator",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="nearest",
        notes="INT8 tensor cores. Most deployed edge GPU platform.",
    )
)

_reg(
    HardwareProfile(
        name="habana_gaudi",
        vendor="Intel",
        family="Habana Gaudi 2/3",
        platform_class="accelerator",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        notes="Data centre AI training. FP8/INT8/BF16 native.",
    )
)

_reg(
    HardwareProfile(
        name="drp_ai",
        vendor="Renesas",
        family="RZ/V2H DRP-AI",
        platform_class="accelerator",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="nearest",
        notes="Automotive AI accelerator. INT8 inference.",
    )
)

# ── Embedded FPGA (eFPGA) IP ────────────────────────────────────────

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

# ── Vision / Sensor Processors ──────────────────────────────────────

_reg(
    HardwareProfile(
        name="imx500",
        vendor="Sony",
        family="IMX500/IMX501",
        platform_class="accelerator",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="AI-on-sensor. Processes spikes at the pixel level.",
    )
)

_reg(
    HardwareProfile(
        name="samsung_npu",
        vendor="Samsung",
        family="Exynos NPU",
        platform_class="accelerator",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        notes="Exynos NPU in billions of phones. INT8/INT16.",
    )
)

# ── CGRA / Coarse-Grained Reconfigurable ─────────────────────────────

_reg(
    HardwareProfile(
        name="samsung_cgra",
        vendor="Samsung",
        family="CGRA-NPU",
        platform_class="cgra",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        dsp_block="PE",
        dsp_mult_a=16,
        dsp_mult_b=16,
        max_freq_mhz=1000,
        notes="Samsung Exynos CGRA: reconfigurable PE array for on-device AI.",
    )
)
_reg(
    HardwareProfile(
        name="qualcomm_npu_cgra",
        vendor="Qualcomm",
        family="NPU-CGRA",
        platform_class="cgra",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="nearest",
        dsp_block="NPU-PE",
        dsp_mult_a=8,
        dsp_mult_b=8,
        max_freq_mhz=1200,
        notes="Qualcomm Hexagon NPU CGRA: reconfigurable accelerator in Snapdragon.",
    )
)
_reg(
    HardwareProfile(
        name="pact_xtensa",
        vendor="Cadence",
        family="Xtensa-CGRA",
        platform_class="cgra",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        dsp_block="TIE",
        dsp_mult_a=16,
        dsp_mult_b=16,
        max_freq_mhz=800,
        notes="Cadence Xtensa CGRA: configurable processor with CGRA extension.",
    )
)

# ── 3D-Stacked / Monolithic 3D ──────────────────────────────────────

_reg(
    HardwareProfile(
        name="tsmc_soic",
        vendor="TSMC",
        family="SoIC-3D",
        platform_class="3d_stacked",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        max_freq_mhz=2000,
        notes="TSMC SoIC: monolithic 3D. Vertical neuron-synapse partitioning.",
    )
)
_reg(
    HardwareProfile(
        name="intel_foveros",
        vendor="Intel",
        family="Foveros-Direct",
        platform_class="3d_stacked",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        max_freq_mhz=1800,
        notes="Intel Foveros Direct: 3D face-to-face bonding for heterogeneous AI.",
    )
)
_reg(
    HardwareProfile(
        name="amd_3dv",
        vendor="AMD",
        family="3D V-Cache",
        platform_class="3d_stacked",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        max_freq_mhz=2200,
        notes="AMD 3D V-Cache: stacked SRAM for massive synaptic weight cache.",
    )
)

# ── Edge MCU / TinyML ────────────────────────────────────────────────

_reg(
    HardwareProfile(
        name="rp2040",
        vendor="Raspberry Pi",
        family="RP2040",
        platform_class="edge_mcu",
        data_width=16,
        fraction=8,
        overflow="wrap",
        rounding="truncate",
        max_freq_mhz=133,
        notes="RP2040: dual Cortex-M0+, 264KB SRAM. TinyML SNN inference.",
    )
)
_reg(
    HardwareProfile(
        name="esp32_s3",
        vendor="Espressif",
        family="ESP32-S3",
        platform_class="edge_mcu",
        data_width=16,
        fraction=8,
        overflow="wrap",
        rounding="truncate",
        max_freq_mhz=240,
        notes="ESP32-S3: dual-core Xtensa + vector extensions. WiFi/BLE SNN.",
    )
)
_reg(
    HardwareProfile(
        name="stm32h7",
        vendor="STMicroelectronics",
        family="STM32H7",
        platform_class="edge_mcu",
        data_width=32,
        fraction=16,
        overflow="saturate",
        rounding="truncate",
        max_freq_mhz=480,
        notes="STM32H7: Cortex-M7 @ 480 MHz. Best-in-class MCU for SNN.",
    )
)
_reg(
    HardwareProfile(
        name="nrf5340",
        vendor="Nordic",
        family="nRF5340",
        platform_class="edge_mcu",
        data_width=16,
        fraction=8,
        overflow="wrap",
        rounding="truncate",
        max_freq_mhz=128,
        notes="nRF5340: dual Cortex-M33. BLE 5.3 + edge SNN inference.",
    )
)
_reg(
    HardwareProfile(
        name="max78000",
        vendor="Analog Devices",
        family="MAX78000",
        platform_class="edge_mcu",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        dsp_block="CNN",
        dsp_mult_a=8,
        dsp_mult_b=8,
        max_freq_mhz=100,
        notes="MAX78000: integrated CNN accelerator. Ultra-low-power edge AI.",
    )
)

# ── RISC-V AI Accelerators ──────────────────────────────────────────

_reg(
    HardwareProfile(
        name="sifive_x280",
        vendor="SiFive",
        family="Intelligence-X280",
        platform_class="accelerator",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        dsp_block="VALU",
        dsp_mult_a=16,
        dsp_mult_b=16,
        max_freq_mhz=2000,
        notes="SiFive X280: RISC-V vector processor. RVV 1.0 for SNN kernels.",
    )
)
_reg(
    HardwareProfile(
        name="qualcomm_ventana",
        vendor="Qualcomm/Ventana",
        family="Veyron-V2",
        platform_class="accelerator",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        max_freq_mhz=3000,
        notes="Qualcomm/Ventana Veyron: high-performance RISC-V for data center AI.",
    )
)
_reg(
    HardwareProfile(
        name="ainekko_rv",
        vendor="Ainekko",
        family="ET-Mirai",
        platform_class="accelerator",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="nearest",
        dsp_block="MINION",
        dsp_mult_a=8,
        dsp_mult_b=8,
        max_freq_mhz=1000,
        notes="Ainekko ET-Mirai: open-source many-core RISC-V (ex-Esperanto).",
    )
)
