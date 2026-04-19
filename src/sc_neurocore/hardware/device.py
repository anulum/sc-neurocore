# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Neuromorphic Hardware Device Catalog

"""Device specifications for neuromorphic hardware targets.

Provides a hardware-agnostic ``DeviceSpec`` dataclass and a catalog
of known neuromorphic platforms with their physical constraints,
derived from published datasheets and technical reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class DeviceFamily(Enum):
    """Supported neuromorphic hardware families."""

    LOIHI = auto()
    LOIHI2 = auto()
    SPINNAKER = auto()
    SPINNAKER2 = auto()
    BRAINSCALES = auto()
    BRAINSCALES2 = auto()
    FPGA_GENERIC = auto()
    AKIDA = auto()


@dataclass(frozen=True)
class DeviceSpec:
    """Physical specification of a neuromorphic device.

    Attributes:
        family: Hardware family identifier.
        cores: Number of neuro-cores on the chip.
        neurons_per_core: Maximum neurons per core.
        synapses_per_core: Maximum synaptic connections per core.
        axons_per_core: Maximum input axons per core.
        tick_ns: Duration of one simulation tick in nanoseconds.
        precision_bits: Weight precision in bits.
        supports_learning: Whether on-chip learning is supported.
        power_per_core_mw: Estimated power per active core (mW).
        max_fan_in: Maximum fan-in per neuron.
        max_fan_out: Maximum fan-out per neuron.
        weight_bits: Synaptic weight bit-width.
        delay_bits: Synaptic delay bit-width.
        max_delay_ticks: Maximum synaptic delay in ticks.
    """

    family: DeviceFamily
    cores: int
    neurons_per_core: int
    synapses_per_core: int
    axons_per_core: int
    tick_ns: float
    precision_bits: int
    supports_learning: bool
    power_per_core_mw: float
    max_fan_in: int = 256
    max_fan_out: int = 4096
    weight_bits: int = 8
    delay_bits: int = 6
    max_delay_ticks: int = 63


# Published device specifications
# Sources:
#   Loihi:       Davies, M. et al. (2018). IEEE Micro 38(1):82–99.
#   Loihi 2:     Orchard, G. et al. (2021). IEEE Micro 41(6):14–20 (Lava).
#   SpiNNaker:   Furber, S.B. et al. (2014). Proc. IEEE 102(5):652–665.
#   SpiNNaker 2: Mayr, C. et al. (2019). IEEE Trans. Biomed. Circuits Syst. 13(5):1001.
#   BrainScaleS:  Schemmel, J. et al. (2010). Proc. ISCAS 2010:1947–1950.
#   BrainScaleS-2: Pehle, C. et al. (2022). Front. Neurosci. 16:795876.
#   Akida:       official product datasheets (BrainChip, 2023).

DEVICE_CATALOG: dict[DeviceFamily, DeviceSpec] = {
    DeviceFamily.LOIHI: DeviceSpec(
        family=DeviceFamily.LOIHI,
        cores=128,
        neurons_per_core=1024,
        synapses_per_core=128 * 1024,
        axons_per_core=4096,
        tick_ns=1000.0,  # 1 µs per tick
        precision_bits=9,
        supports_learning=True,
        power_per_core_mw=0.15,
        max_fan_in=4096,
        max_fan_out=4096,
        weight_bits=9,
        delay_bits=6,
        max_delay_ticks=63,
    ),
    DeviceFamily.LOIHI2: DeviceSpec(
        family=DeviceFamily.LOIHI2,
        cores=128,
        neurons_per_core=8192,
        synapses_per_core=128 * 1024,
        axons_per_core=8192,
        tick_ns=500.0,
        precision_bits=8,
        supports_learning=True,
        power_per_core_mw=0.10,
        max_fan_in=8192,
        max_fan_out=8192,
        weight_bits=8,
        delay_bits=6,
        max_delay_ticks=63,
    ),
    DeviceFamily.SPINNAKER: DeviceSpec(
        family=DeviceFamily.SPINNAKER,
        cores=18,  # 18 ARM968 per chip
        neurons_per_core=256,  # typical for LIF
        synapses_per_core=16384,
        axons_per_core=256,
        tick_ns=1_000_000.0,  # 1 ms biological tick
        precision_bits=16,
        supports_learning=True,
        power_per_core_mw=1.0,
        max_fan_in=256,
        max_fan_out=1024,
        weight_bits=16,
        delay_bits=4,
        max_delay_ticks=15,
    ),
    DeviceFamily.SPINNAKER2: DeviceSpec(
        family=DeviceFamily.SPINNAKER2,
        cores=152,
        neurons_per_core=512,
        synapses_per_core=65536,
        axons_per_core=512,
        tick_ns=500_000.0,
        precision_bits=32,
        supports_learning=True,
        power_per_core_mw=0.5,
        max_fan_in=512,
        max_fan_out=2048,
        weight_bits=16,
        delay_bits=8,
        max_delay_ticks=255,
    ),
    DeviceFamily.BRAINSCALES: DeviceSpec(
        family=DeviceFamily.BRAINSCALES,
        cores=1,  # wafer-scale, single "core" abstraction
        neurons_per_core=512,
        synapses_per_core=114688,
        axons_per_core=256,
        tick_ns=10.0,  # 10⁴x speedup → 10 ns effective
        precision_bits=4,
        supports_learning=False,
        power_per_core_mw=5.0,
        max_fan_in=224,
        max_fan_out=256,
        weight_bits=4,
        delay_bits=0,
        max_delay_ticks=0,
    ),
    DeviceFamily.BRAINSCALES2: DeviceSpec(
        family=DeviceFamily.BRAINSCALES2,
        cores=2,  # 2 PPUs per HICANN-X
        neurons_per_core=256,
        synapses_per_core=65536,
        axons_per_core=256,
        tick_ns=10.0,
        precision_bits=6,
        supports_learning=True,
        power_per_core_mw=3.0,
        max_fan_in=256,
        max_fan_out=256,
        weight_bits=6,
        delay_bits=0,
        max_delay_ticks=0,
    ),
    DeviceFamily.FPGA_GENERIC: DeviceSpec(
        family=DeviceFamily.FPGA_GENERIC,
        cores=64,  # configurable
        neurons_per_core=512,
        synapses_per_core=32768,
        axons_per_core=512,
        tick_ns=100.0,
        precision_bits=16,
        supports_learning=True,
        power_per_core_mw=2.0,
        max_fan_in=512,
        max_fan_out=512,
        weight_bits=16,
        delay_bits=8,
        max_delay_ticks=255,
    ),
    DeviceFamily.AKIDA: DeviceSpec(
        family=DeviceFamily.AKIDA,
        cores=80,
        neurons_per_core=256,
        synapses_per_core=65536,
        axons_per_core=256,
        tick_ns=1000.0,
        precision_bits=4,
        supports_learning=True,
        power_per_core_mw=0.05,
        max_fan_in=256,
        max_fan_out=4096,
        weight_bits=4,
        delay_bits=0,
        max_delay_ticks=0,
    ),
}


def get_device(family: DeviceFamily | str) -> DeviceSpec:
    """Look up a device specification by family name or enum."""
    if isinstance(family, str):
        family = DeviceFamily[family.upper()]
    spec = DEVICE_CATALOG.get(family)
    if spec is None:
        raise ValueError(f"Unknown device family: {family}")
    return spec
