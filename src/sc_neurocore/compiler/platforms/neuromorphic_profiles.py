# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore

"""Register neuromorphic and event-driven hardware profiles."""

from __future__ import annotations

from .registry import HardwareProfile, _reg

# ── Neuromorphic Chips ───────────────────────────────────────────────

_reg(
    HardwareProfile(
        name="loihi2",
        vendor="Intel",
        family="Loihi 2",
        platform_class="neuromorphic",
        data_width=24,
        fraction=12,
        overflow="wrap",
        rounding="truncate",
        notes="24-bit membrane potential. Wrap on overflow (hardware behaviour).",
    )
)

_reg(
    HardwareProfile(
        name="truenorth",
        vendor="IBM",
        family="TrueNorth",
        platform_class="neuromorphic",
        data_width=8,
        fraction=7,
        overflow="saturate",
        rounding="truncate",
        notes="1-bit stochastic neurons. Q1.7 approximation for parameter transfer.",
    )
)

_reg(
    HardwareProfile(
        name="brainscales2",
        vendor="Heidelberg",
        family="BrainScaleS-2",
        platform_class="neuromorphic",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="nearest",
        notes="Analog VLSI with 8-bit DAC/ADC. Q4.4 for digital calibration.",
    )
)

_reg(
    HardwareProfile(
        name="spinnaker2",
        vendor="TU Dresden",
        family="SpiNNaker 2",
        platform_class="neuromorphic",
        data_width=16,
        fraction=15,
        overflow="saturate",
        rounding="nearest",
        notes="ARM Cortex-M4 with CMSIS-DSP. Q1.15 is the native format.",
    )
)

_reg(
    HardwareProfile(
        name="akida",
        vendor="BrainChip",
        family="Akida 2.0",
        platform_class="neuromorphic",
        data_width=8,
        fraction=7,
        overflow="saturate",
        rounding="truncate",
        notes="Event-driven neural processor. 8-bit weights.",
    )
)

_reg(
    HardwareProfile(
        name="dynap_se2",
        vendor="SynSense",
        family="DYNAP-SE2",
        platform_class="neuromorphic",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="truncate",
        notes="Mixed-signal neuromorphic. 16-bit digital membrane.",
    )
)

_reg(
    HardwareProfile(
        name="xylo",
        vendor="SynSense",
        family="Xylo",
        platform_class="neuromorphic",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="truncate",
        notes="Digital spiking neural network processor. 16-bit.",
    )
)

# ── Neuromorphic: Next-Generation (2025-2026) ───────────────────────
_reg(
    HardwareProfile(
        name="loihi3",
        vendor="Intel",
        family="Loihi 3",
        platform_class="neuromorphic",
        data_width=32,
        fraction=16,
        overflow="wrap",
        rounding="truncate",
        notes="Loihi 3 (4nm, 8M neurons). 32-bit state, wrap overflow.",
    )
)
_reg(
    HardwareProfile(
        name="northpole",
        vendor="IBM",
        family="NorthPole",
        platform_class="neuromorphic",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="nearest",
        notes="IBM NorthPole: 256-core digital, no DRAM. INT2/INT4/INT8.",
    )
)
_reg(
    HardwareProfile(
        name="innatera_pulsar",
        vendor="Innatera",
        family="Pulsar",
        platform_class="neuromorphic",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="Innatera Pulsar neuromorphic μC. Analog-digital hybrid.",
    )
)

# ── FPGA: Missing Families ──────────────────────────────────────────
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

# ── AI Accelerators: Major Missing ──────────────────────────────────
_reg(
    HardwareProfile(
        name="nvidia_dla",
        vendor="NVIDIA",
        family="DLA (Orin)",
        platform_class="accelerator",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="nearest",
        notes="NVIDIA DLA: dedicated INT8 inference engine in Orin SoC.",
    )
)
_reg(
    HardwareProfile(
        name="mediatek_apu",
        vendor="MediaTek",
        family="APU 790",
        platform_class="accelerator",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="nearest",
        notes="MediaTek APU: ~40% mobile AI market. INT4/INT8/INT16.",
    )
)
_reg(
    HardwareProfile(
        name="aws_inferentia",
        vendor="AWS",
        family="Inferentia2",
        platform_class="accelerator",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        notes="AWS Inferentia2/Trainium2. BF16/FP16/INT8 cloud inference.",
    )
)
_reg(
    HardwareProfile(
        name="qualcomm_nsp",
        vendor="Qualcomm",
        family="Neural Signal Processor",
        platform_class="accelerator",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="nearest",
        notes="Qualcomm NSP: dedicated INT8 inference engine (not HVX DSP).",
    )
)
_reg(
    HardwareProfile(
        name="sambanova",
        vendor="SambaNova",
        family="RDU",
        platform_class="accelerator",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        notes="SambaNova RDU: reconfigurable dataflow architecture. BF16/FP16.",
    )
)
_reg(
    HardwareProfile(
        name="cambricon_mlu",
        vendor="Cambricon",
        family="MLU370",
        platform_class="accelerator",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        notes="Cambricon MLU370/590: dominant Chinese AI silicon. INT8/FP16.",
    )
)

# ── Emerging Compute Paradigms ──────────────────────────────────────
_reg(
    HardwareProfile(
        name="superconducting",
        vendor="Various",
        family="AQFP/SFQ",
        platform_class="emerging",
        data_width=8,
        fraction=4,
        overflow="wrap",
        rounding="truncate",
        notes="Superconducting AQFP/SFQ: ~100 GHz, μW power. Post-CMOS.",
    )
)
_reg(
    HardwareProfile(
        name="cim_sram",
        vendor="Various",
        family="CIM-SRAM",
        platform_class="emerging",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="Compute-in-SRAM (TSMC/Samsung): analog MAC in memory cell.",
    )
)
_reg(
    HardwareProfile(
        name="analog_ai",
        vendor="IBM/Mythic",
        family="PCM/ReRAM",
        platform_class="emerging",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="Phase-change / ReRAM analog compute: 0.1 TOPS/W class.",
    )
)
_reg(
    HardwareProfile(
        name="event_camera",
        vendor="Prophesee/Sony",
        family="DVS/IMX636",
        platform_class="emerging",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="truncate",
        notes="DVS event-camera: natural AER spike interface. IMX636 native.",
    )
)

# ── Photonic / Optical Compute ──────────────────────────────────────
_reg(
    HardwareProfile(
        name="lightmatter_passage",
        vendor="Lightmatter",
        family="Passage",
        platform_class="photonic",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        dsp_block="MZI",
        dsp_mult_a=8,
        dsp_mult_b=8,
        notes="Lightmatter Passage: photonic MZI matrix multiply. Phase-shift weights.",
    )
)
_reg(
    HardwareProfile(
        name="lightelligence_pace",
        vendor="Lightelligence",
        family="PACE2",
        platform_class="photonic",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        dsp_block="OE-MAC",
        dsp_mult_a=8,
        dsp_mult_b=8,
        notes="Lightelligence PACE2: hybrid opto-electronic accelerator. IPO 2026.",
    )
)
_reg(
    HardwareProfile(
        name="xanadu_x8",
        vendor="Xanadu",
        family="X8/Borealis",
        platform_class="photonic",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        dsp_block="GBS",
        dsp_mult_a=8,
        dsp_mult_b=8,
        notes="Xanadu X8/Borealis: photonic quantum-classical hybrid.",
    )
)
_reg(
    HardwareProfile(
        name="ipronics_smartlight",
        vendor="iPronics",
        family="SmartLight",
        platform_class="photonic",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        dsp_block="MZI",
        dsp_mult_a=8,
        dsp_mult_b=8,
        notes="iPronics SmartLight: programmable photonic processor. EU-funded.",
    )
)
_reg(
    HardwareProfile(
        name="luminous_computing",
        vendor="Luminous",
        family="Photonic Engine",
        platform_class="photonic",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        dsp_block="MZI",
        dsp_mult_a=8,
        dsp_mult_b=8,
        notes="Luminous Computing: optical interconnect + compute co-design.",
    )
)

# ── Chiplet / UCIe / Heterogeneous Integration ──────────────────────
_reg(
    HardwareProfile(
        name="tenstorrent_blackhole",
        vendor="Tenstorrent",
        family="Blackhole",
        platform_class="accelerator",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        dsp_block="Tensix",
        dsp_mult_a=16,
        dsp_mult_b=16,
        notes="Tenstorrent Blackhole: RISC-V + AI chiplet. Galaxy platform 2026.",
    )
)
_reg(
    HardwareProfile(
        name="cerebras_wse3",
        vendor="Cerebras",
        family="WSE-3",
        platform_class="accelerator",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        dsp_block="PE",
        dsp_mult_a=16,
        dsp_mult_b=16,
        max_freq_mhz=1000,
        notes="Cerebras WSE-3: wafer-scale engine for large-scale AI accelerator deployments.",
    )
)
_reg(
    HardwareProfile(
        name="intel_ponte_vecchio",
        vendor="Intel",
        family="Ponte Vecchio",
        platform_class="accelerator",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        dsp_block="XMX",
        dsp_mult_a=16,
        dsp_mult_b=16,
        notes="Intel Ponte Vecchio: multi-tile GPU, UCIe interconnect.",
    )
)
_reg(
    HardwareProfile(
        name="amd_mi300x",
        vendor="AMD",
        family="Instinct MI300X",
        platform_class="accelerator",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        dsp_block="CDNA3",
        dsp_mult_a=16,
        dsp_mult_b=16,
        notes="AMD MI300X: hybrid CPU+GPU+HBM chiplet. 3D-stacked.",
    )
)
_reg(
    HardwareProfile(
        name="ucie_generic",
        vendor="UCIe Consortium",
        family="UCIe 1.1",
        platform_class="accelerator",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        notes="Universal Chiplet Interconnect Express generic template.",
    )
)

# ── Processing-in-Memory / CXL ──────────────────────────────────────
_reg(
    HardwareProfile(
        name="upmem_pim",
        vendor="UPMEM",
        family="PIM-DRAM",
        platform_class="in_memory",
        data_width=32,
        fraction=16,
        overflow="saturate",
        rounding="truncate",
        notes="UPMEM PIM-DRAM: 2560 DPUs per module. In-DRAM processing.",
    )
)
_reg(
    HardwareProfile(
        name="samsung_hbm_pim",
        vendor="Samsung",
        family="HBM-PIM",
        platform_class="in_memory",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        notes="Samsung HBM-PIM: in-bank compute for attention/embedding.",
    )
)
_reg(
    HardwareProfile(
        name="sk_hynix_aim",
        vendor="SK Hynix",
        family="AIM",
        platform_class="in_memory",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        notes="SK Hynix Accelerator-in-Memory: GDDR6-AiM for CXL.",
    )
)
_reg(
    HardwareProfile(
        name="cxl_type3",
        vendor="CXL Consortium",
        family="CXL 3.0 Type-3",
        platform_class="in_memory",
        data_width=32,
        fraction=16,
        overflow="saturate",
        rounding="nearest",
        notes="CXL 3.0 Type-3 memory expander: shared memory pooling.",
    )
)
_reg(
    HardwareProfile(
        name="axdimm",
        vendor="Samsung",
        family="AxDIMM",
        platform_class="in_memory",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="truncate",
        notes="Samsung AxDIMM: near-DRAM processing for embedding lookups.",
    )
)

# ── Next-Gen Neuromorphic ───────────────────────────────────────────
_reg(
    HardwareProfile(
        name="akida2",
        vendor="BrainChip",
        family="Akida 2 / AKD1500",
        platform_class="neuromorphic",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="BrainChip Akida 2: event-based, 1/4/8-bit quant, on-chip learning.",
    )
)
_reg(
    HardwareProfile(
        name="spinnaker2",
        vendor="SpiNNcloud",
        family="SpiNNaker 2",
        platform_class="neuromorphic",
        data_width=32,
        fraction=16,
        overflow="wrap",
        rounding="truncate",
        notes="SpiNNaker 2: ARM-based massively parallel GALS. Brain-scale sim.",
    )
)
_reg(
    HardwareProfile(
        name="dynapse2",
        vendor="SynSense",
        family="DYNAP-SE2",
        platform_class="neuromorphic",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="truncate",
        notes="SynSense DYNAP-SE2: mixed-signal analog neuron circuits.",
    )
)
_reg(
    HardwareProfile(
        name="rain_neuromorphic",
        vendor="Rain AI",
        family="Rain NeuralCore",
        platform_class="neuromorphic",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="Rain AI: memristive crossbar architecture. $100M+ funded.",
    )
)
_reg(
    HardwareProfile(
        name="brainscales2",
        vendor="Heidelberg",
        family="BrainScaleS-2",
        platform_class="neuromorphic",
        data_width=8,
        fraction=4,
        overflow="wrap",
        rounding="truncate",
        notes="BrainScaleS-2: analog accelerated neuro, 1000× bio realtime.",
    )
)

# ── Sovereign / Defence / Aerospace ─────────────────────────────────
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

# ── Automotive / Edge AI SoC ────────────────────────────────────────
_reg(
    HardwareProfile(
        name="mythic_m1076",
        vendor="Mythic",
        family="M1076",
        platform_class="accelerator",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="Mythic M1076: analog CIM. Honda automotive partnership 2026.",
    )
)
_reg(
    HardwareProfile(
        name="mobileye_eyeq6",
        vendor="Mobileye",
        family="EyeQ6",
        platform_class="accelerator",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="nearest",
        dsp_block="VMP",
        dsp_mult_a=8,
        dsp_mult_b=8,
        notes="Mobileye EyeQ6: 90 TOPS automotive vision SoC. INT8.",
    )
)
_reg(
    HardwareProfile(
        name="horizon_j6",
        vendor="Horizon Robotics",
        family="Journey 6",
        platform_class="accelerator",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="nearest",
        dsp_block="BPU",
        dsp_mult_a=8,
        dsp_mult_b=8,
        notes="Horizon J6: BPU neural accelerator. Chinese auto market leader.",
    )
)
_reg(
    HardwareProfile(
        name="ambarella_cv72s",
        vendor="Ambarella",
        family="CV72S",
        platform_class="accelerator",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="nearest",
        dsp_block="CVflow",
        dsp_mult_a=8,
        dsp_mult_b=8,
        notes="Ambarella CV72S: edge AI SoC. Automotive + security cameras.",
    )
)
_reg(
    HardwareProfile(
        name="hailo15",
        vendor="Hailo",
        family="Hailo-15",
        platform_class="accelerator",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="nearest",
        dsp_block="Structure",
        dsp_mult_a=8,
        dsp_mult_b=8,
        notes="Hailo-15: next-gen edge AI. 20 TOPS/W, vision processor.",
    )
)
_reg(
    HardwareProfile(
        name="syntiant_ndp120",
        vendor="Syntiant",
        family="NDP120",
        platform_class="neuromorphic",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="Syntiant NDP120: ultra-low-power NDP, always-on audio/sensor.",
    )
)

# ── Spintronic / MRAM ────────────────────────────────────────────────

_reg(
    HardwareProfile(
        name="everspin_stt_mram",
        vendor="Everspin",
        family="STT-MRAM",
        platform_class="spintronic",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="Everspin STT-MRAM: non-volatile synaptic weight store. Zero standby.",
    )
)
_reg(
    HardwareProfile(
        name="samsung_sot_mram",
        vendor="Samsung",
        family="SOT-MRAM",
        platform_class="spintronic",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="Samsung SOT-MRAM: spin-orbit torque. Faster write, lower endurance.",
    )
)

# ── Ferroelectric CiM ───────────────────────────────────────────────

_reg(
    HardwareProfile(
        name="gf_fefet",
        vendor="GlobalFoundries",
        family="FeFET-22FDX",
        platform_class="ferroelectric",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="GF 22FDX FeFET: HfO₂ ferroelectric compute-in-memory. MLC capable.",
    )
)
_reg(
    HardwareProfile(
        name="sk_hynix_feram",
        vendor="SK Hynix",
        family="FeRAM",
        platform_class="ferroelectric",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="SK Hynix FeRAM: ferroelectric RAM for non-volatile neural state.",
    )
)

# ── Analog Mixed-Signal ──────────────────────────────────────────────

_reg(
    HardwareProfile(
        name="aspinity_aml100",
        vendor="Aspinity",
        family="AML100",
        platform_class="analog_mixed",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="Aspinity AML100: analog ML at the sensor. Always-on anomaly detection, µW power.",
    )
)
_reg(
    HardwareProfile(
        name="renesas_analog_ai",
        vendor="Renesas",
        family="AnalogAI",
        platform_class="analog_mixed",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="Renesas Analog AI: mixed-signal compute-at-ADC boundary. "
        "Industrial IoT edge inference.",
    )
)

# ── RRAM / Memristive Crossbar ───────────────────────────────────────

_reg(
    HardwareProfile(
        name="weebit_reram",
        vendor="Weebit Nano",
        family="ReRAM-ACiM",
        platform_class="rram",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="Weebit Nano ReRAM: analog compute-in-memory crossbar. "
        "200 TOPS/W, licensed to TI/Onsemi.",
    )
)
_reg(
    HardwareProfile(
        name="crossbar_rram",
        vendor="Crossbar",
        family="ReRAM-1T1R",
        platform_class="rram",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="Crossbar Inc RRAM: 1T1R selector. Non-volatile weights, multi-level cell synapses.",
    )
)
_reg(
    HardwareProfile(
        name="adesto_cbram",
        vendor="Adesto",
        family="CBRAM",
        platform_class="rram",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="Adesto CBRAM: conductive-bridge RAM. Low-power IoT NVM.",
    )
)

# ── SRAM Compute-in-Memory ───────────────────────────────────────────

_reg(
    HardwareProfile(
        name="tsmc_cim_n7",
        vendor="TSMC",
        family="CIM-N7",
        platform_class="sram_cim",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        dsp_block="CIM_MACRO",
        dsp_mult_a=8,
        dsp_mult_b=8,
        max_freq_mhz=1000,
        notes="TSMC N7 SRAM-CIM macro: digital compute-in-memory. "
        "Standard cell integration, foundry-qualified.",
    )
)
_reg(
    HardwareProfile(
        name="samsung_cim_sf3",
        vendor="Samsung",
        family="CIM-SF3",
        platform_class="sram_cim",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        dsp_block="CIM_MACRO",
        dsp_mult_a=8,
        dsp_mult_b=8,
        max_freq_mhz=900,
        notes="Samsung SF3 SRAM-CIM: 3nm GAA compute-in-memory macro.",
    )
)

# ── Cryogenic CMOS ───────────────────────────────────────────────────

_reg(
    HardwareProfile(
        name="intel_horse_ridge",
        vendor="Intel",
        family="Horse-Ridge-II",
        platform_class="cryo_cmos",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        max_freq_mhz=6000,
        notes="Intel Horse Ridge II: cryogenic CMOS controller at 4K. "
        "Quantum-classical interface for neuromorphic control.",
    )
)
_reg(
    HardwareProfile(
        name="google_cryo_ctrl",
        vendor="Google",
        family="Cryo-Controller",
        platform_class="cryo_cmos",
        data_width=16,
        fraction=8,
        overflow="saturate",
        rounding="nearest",
        max_freq_mhz=4000,
        notes="Google cryogenic controller: CMOS at 4K for quantum chip readout and control.",
    )
)

# ── DNA / Molecular ──────────────────────────────────────────────────

_reg(
    HardwareProfile(
        name="microsoft_dna_store",
        vendor="Microsoft",
        family="DNA-Storage",
        platform_class="dna_molecular",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="Microsoft DNA Storage: enzymatic DNA synthesis for archival "
        "compute. Output = nucleotide sequence.",
    )
)
_reg(
    HardwareProfile(
        name="asu_dna_perovskite",
        vendor="ASU",
        family="DNA-Perovskite",
        platform_class="dna_molecular",
        data_width=8,
        fraction=4,
        overflow="saturate",
        rounding="truncate",
        notes="ASU DNA-Perovskite: bio-hybrid synaptic device. DNA base-pair "
        "gated perovskite semiconductor.",
    )
)
