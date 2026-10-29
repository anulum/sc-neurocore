# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Registered synthesis target profiles

"""Own FPGA synthesis profiles and copy-returning registration metadata."""

from __future__ import annotations

from dataclasses import dataclass

_TARGETS: dict[str, dict[str, str | None]] = {
    "ice40": {"synth_cmd": "synth_ice40", "pnr": "nextpnr-ice40", "device": "up5k"},
    "ecp5": {
        "synth_cmd": "synth_ecp5",
        "pnr": "nextpnr-ecp5",
        "device": "25k",
        "package": "CABGA381",
    },
    "gowin": {"synth_cmd": "synth_gowin", "pnr": None, "device": None},
    "xilinx": {"synth_cmd": "synth_xilinx -flatten", "pnr": None, "device": None},
}

#: Resource capacity of the device each target is judged against, and that
#: device's name. Gowin and Xilinx synthesis here is not device-bound, so the
#: device is named with the verdict rather than implied by the family.
#:
#: - iCE40 UltraPlus UP5K: 5280 LUT4 and flip-flops, 30 EBR, 8 SB_MAC16
#:   (Lattice FPGA-DS-02008; nextpnr-ice40 --up5k reports the same).
#: - ECP5 LFE5U-25F: 24288 LUT4 and flip-flops, 56 DP16KD, 28 MULT18X18D, as
#:   nextpnr-ecp5 --25k reports them (the data sheet rounds to "24K"). The row
#:   said 24576.
#: - Gowin GW2A-18: 20736 LUT4, 15552 flip-flops, 46 B-SRAM, 48 18x18
#:   multipliers (Gowin DS102). The row said 20736 flip-flops, 41 B-SRAM and
#:   no multipliers, matching no device.
#: - Artix-7 XC7A35T: 20800 LUT6, 41600 flip-flops, 50 RAMB36 (counted as 100
#:   RAMB18 halves, so an 18 Kb block counts once), 90 DSP48E1 (Xilinx DS180).
_DEVICE_CAPACITY = {
    "ice40": {"luts": 5280, "ffs": 5280, "brams": 30, "dsps": 8},
    "ecp5": {"luts": 24288, "ffs": 24288, "brams": 56, "dsps": 28},
    "gowin": {"luts": 20736, "ffs": 15552, "brams": 46, "dsps": 48},
    "xilinx": {"luts": 20800, "ffs": 41600, "brams": 100, "dsps": 90},
}

_CAPACITY_DEVICE = {
    "ice40": "iCE40 UP5K",
    "ecp5": "ECP5 LFE5U-25F",
    "gowin": "Gowin GW2A-18",
    "xilinx": "Artix-7 XC7A35T",
}


@dataclass(frozen=True, slots=True)
class SynthesisTargetProfile:
    """Registered synthesis configuration, without a run or board verdict.

    Capacity names a reference device. A missing device selector means the
    synthesis command is not bound to that device. The returned capacity is
    a copy, so catalogue readers cannot mutate the synthesis registry.
    """

    target: str
    capacity_device: str
    capacity: dict[str, int]
    synthesis_command: str | None
    routing_tool: str | None
    device_selector: str | None
    package_selector: str | None


def synthesis_target_profiles() -> tuple[SynthesisTargetProfile, ...]:
    """Describe registered targets without probing tools or running synthesis.

    Returns
    -------
    tuple of SynthesisTargetProfile
        Independent snapshots in the same order as ``supported_targets``.
        Registration and reference capacity do not attest tool availability,
        compilation, timing closure, board execution or measured power.
    """
    return tuple(
        SynthesisTargetProfile(
            target=target,
            capacity_device=_CAPACITY_DEVICE[target],
            capacity=dict(_DEVICE_CAPACITY[target]),
            synthesis_command=config["synth_cmd"],
            routing_tool=config["pnr"],
            device_selector=config["device"],
            package_selector=config.get("package"),
        )
        for target, config in _TARGETS.items()
    )
