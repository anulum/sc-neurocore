# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — embedded processor profile registrations

"""Register DSP and edge-microcontroller hardware profiles."""

from __future__ import annotations

from .registry import HardwareProfile, _reg

# ── DSP Processors ──────────────────────────────────────────────────


def _register_dsp_profiles() -> None:
    """Register digital signal processor profiles."""
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


# ── Edge MCU / TinyML ────────────────────────────────────────────────


def _register_edge_mcu_profiles() -> None:
    """Register edge microcontroller and TinyML profiles."""
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
            notes="STM32H7: Cortex-M7 @ 480 MHz. MCU profile for edge SNN.",
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
