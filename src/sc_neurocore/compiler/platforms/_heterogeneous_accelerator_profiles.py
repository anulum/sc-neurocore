# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — heterogeneous accelerator profile registrations

"""Register heterogeneous, chiplet, automotive, and edge accelerators."""

from __future__ import annotations

from .registry import HardwareProfile, _reg


def _register_ai_accelerator_profiles() -> None:
    """Register dedicated artificial-intelligence accelerator profiles."""
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


def _register_chiplet_accelerator_profiles() -> None:
    """Register chiplet and heterogeneous-integration accelerator profiles."""
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


def _register_automotive_edge_profiles() -> None:
    """Register automotive and edge-system accelerator profiles."""
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
