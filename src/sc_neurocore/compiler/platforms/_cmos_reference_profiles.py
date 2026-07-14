# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ASIC and simulation profile registrations

"""Register generic ASIC targets and fixed-point simulation references."""

from __future__ import annotations

from .registry import HardwareProfile, _reg

# ── ASIC Targets ─────────────────────────────────────────────────────


def _register_asic_profiles() -> None:
    """Register generic ASIC target profiles."""
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


def _register_simulation_profiles() -> None:
    """Register fixed-point simulation reference profiles."""
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
