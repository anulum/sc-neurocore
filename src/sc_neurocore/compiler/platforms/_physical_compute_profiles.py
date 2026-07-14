# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — physical-compute hardware-profile registrations

"""Register optical, analogue, cryogenic, and molecular profiles."""

from __future__ import annotations

from .registry import HardwareProfile, _reg


def _register_emerging_compute_profiles() -> None:
    """Register emerging physical-compute paradigm profiles."""
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


def _register_photonic_compute_profiles() -> None:
    """Register photonic and optical-compute profiles."""
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


def _register_analog_mixed_signal_profiles() -> None:
    """Register analogue mixed-signal profiles."""
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


def _register_cryogenic_cmos_profiles() -> None:
    """Register cryogenic CMOS control profiles."""
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


def _register_molecular_profiles() -> None:
    """Register DNA and molecular-compute profiles."""
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
