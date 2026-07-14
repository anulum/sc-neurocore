# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — conventional accelerator profile registrations

"""Register AI, edge, vision, and RISC-V accelerator profiles."""

from __future__ import annotations

from .registry import HardwareProfile, _reg

# ── AI / ML Accelerators ────────────────────────────────────────────


def _register_ai_accelerator_profiles() -> None:
    """Register AI and machine-learning accelerator profiles."""
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


# ── Edge AI Accelerators ────────────────────────────────────────────


def _register_edge_ai_accelerator_profiles() -> None:
    """Register edge AI accelerator profiles."""
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


# ── Vision / Sensor Processors ──────────────────────────────────────


def _register_vision_sensor_profiles() -> None:
    """Register vision and sensor processor profiles."""
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


# ── RISC-V AI Accelerators ──────────────────────────────────────────


def _register_riscv_ai_accelerator_profiles() -> None:
    """Register RISC-V AI accelerator profiles."""
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
