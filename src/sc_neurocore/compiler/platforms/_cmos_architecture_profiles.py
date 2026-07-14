# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — compute architecture profile registrations

"""Register emerging, CGRA, and 3D-stacked compute profiles."""

from __future__ import annotations

from .registry import HardwareProfile, _reg

# ── Emerging Compute Paradigms ───────────────────────────────────────


def _register_emerging_compute_profiles() -> None:
    """Register emerging compute-architecture profiles."""
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


# ── CGRA / Coarse-Grained Reconfigurable ─────────────────────────────


def _register_cgra_profiles() -> None:
    """Register coarse-grained reconfigurable architecture profiles."""
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


def _register_stacked_3d_profiles() -> None:
    """Register 3D-stacked integration profiles."""
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
