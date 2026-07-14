# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — memory-compute hardware-profile registrations

"""Register processing-in-memory and non-volatile compute profiles."""

from __future__ import annotations

from .registry import HardwareProfile, _reg


def _register_processing_in_memory_profiles() -> None:
    """Register processing-in-memory and CXL profiles."""
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


def _register_spintronic_profiles() -> None:
    """Register spintronic and magnetic-memory profiles."""
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


def _register_ferroelectric_profiles() -> None:
    """Register ferroelectric compute-in-memory profiles."""
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


def _register_rram_profiles() -> None:
    """Register resistive-memory and memristive crossbar profiles."""
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


def _register_sram_cim_profiles() -> None:
    """Register SRAM compute-in-memory profiles."""
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
