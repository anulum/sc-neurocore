# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — event-driven hardware-profile registrations

"""Register event-driven and neuromorphic hardware profiles."""

from __future__ import annotations

from .registry import HardwareProfile, _reg


def _register_neuromorphic_chip_profiles() -> None:
    """Register the established neuromorphic chip profiles."""
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


def _register_recent_neuromorphic_profiles() -> None:
    """Register the recent neuromorphic chip profiles."""
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


def _register_additional_neuromorphic_profiles() -> None:
    """Register the additional neuromorphic architecture profiles."""
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
