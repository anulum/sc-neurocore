# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Neuromorphic chip specification model

"""Chip specification model for multi-target compilation.

Each neuromorphic chip is described by a ChipSpec: cores, neurons/core,
weight precision, connectivity constraints, supported neuron models,
and on-chip learning capabilities. Built-in specs for Loihi 2, SynSense
Xylo/Speck, BrainChip Akida, and SpiNNaker2.

Specs can be loaded from YAML for custom/future chips.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import json


@dataclass
class CoreSpec:
    """Specification for one neuromorphic core."""

    max_neurons: int
    max_synapses_per_neuron: int
    weight_bits: int
    supported_neuron_types: list[str]
    has_on_chip_learning: bool = False
    learning_rules: list[str] = field(default_factory=list)
    max_delay_steps: int = 0


@dataclass
class ChipSpec:
    """Full neuromorphic chip specification.

    Parameters
    ----------
    name : str
        Chip identifier (e.g., 'loihi2', 'xylo', 'akida').
    vendor : str
    total_cores : int
    core : CoreSpec
        Per-core specification (assumes homogeneous cores).
    clock_mhz : float
    power_mw_per_core : float
        Estimated dynamic power per active core.
    routing_topology : str
        'mesh', 'crossbar', 'tree', 'ring'
    max_fan_out : int
        Maximum outgoing connections per neuron.
    analog_noise_cv : float
        Coefficient of variation for analog process variation.
        0.0 for fully digital chips.
    """

    name: str
    vendor: str
    total_cores: int
    core: CoreSpec
    clock_mhz: float = 100.0
    power_mw_per_core: float = 1.0
    routing_topology: str = "mesh"
    max_fan_out: int = 4096
    analog_noise_cv: float = 0.0

    @property
    def total_neurons(self) -> int:
        return self.total_cores * self.core.max_neurons

    @property
    def total_power_mw(self) -> float:
        return self.total_cores * self.power_mw_per_core

    def fits(self, n_neurons: int, max_fan_out: int = 0) -> bool:
        """Check if a network fits on this chip."""
        if n_neurons > self.total_neurons:
            return False
        return max_fan_out <= self.max_fan_out

    def cores_needed(self, n_neurons: int) -> int:
        """Minimum cores needed for N neurons."""
        return max(1, -(-n_neurons // self.core.max_neurons))  # ceil division


def load_chip_spec(path: str | Path) -> ChipSpec:
    """Load a chip spec from a JSON file."""
    with open(path) as f:
        data = json.load(f)

    core_data = data.pop("core")
    core = CoreSpec(**core_data)
    return ChipSpec(core=core, **data)


# Built-in chip specifications
BUILTIN_CHIPS: dict[str, ChipSpec] = {
    "loihi2": ChipSpec(
        name="loihi2",
        vendor="Intel",
        total_cores=128,
        core=CoreSpec(
            max_neurons=128,
            max_synapses_per_neuron=8192,
            weight_bits=8,
            supported_neuron_types=["LIF", "ALIF", "Izhikevich", "Compartmental"],
            has_on_chip_learning=True,
            learning_rules=["STDP", "R-STDP", "e-prop"],
            max_delay_steps=63,
        ),
        clock_mhz=100,
        power_mw_per_core=0.5,
        routing_topology="mesh",
        max_fan_out=8192,
        analog_noise_cv=0.0,  # digital
    ),
    "xylo": ChipSpec(
        name="xylo",
        vendor="SynSense",
        total_cores=1,
        core=CoreSpec(
            max_neurons=1000,
            max_synapses_per_neuron=1000,
            weight_bits=8,
            supported_neuron_types=["IAF", "LIF"],
            has_on_chip_learning=False,
            max_delay_steps=15,
        ),
        clock_mhz=50,
        power_mw_per_core=0.1,
        routing_topology="crossbar",
        max_fan_out=1000,
        analog_noise_cv=0.0,
    ),
    "speck": ChipSpec(
        name="speck",
        vendor="SynSense",
        total_cores=1,
        core=CoreSpec(
            max_neurons=32768,
            max_synapses_per_neuron=512,
            weight_bits=4,
            supported_neuron_types=["IAF"],
            has_on_chip_learning=False,
            max_delay_steps=0,
        ),
        clock_mhz=200,
        power_mw_per_core=0.5,
        routing_topology="crossbar",
        max_fan_out=512,
        analog_noise_cv=0.0,
    ),
    "akida": ChipSpec(
        name="akida",
        vendor="BrainChip",
        total_cores=80,
        core=CoreSpec(
            max_neurons=256,
            max_synapses_per_neuron=4096,
            weight_bits=4,
            supported_neuron_types=["IF", "LIF"],
            has_on_chip_learning=True,
            learning_rules=["STDP"],
            max_delay_steps=0,
        ),
        clock_mhz=300,
        power_mw_per_core=0.3,
        routing_topology="mesh",
        max_fan_out=4096,
        analog_noise_cv=0.0,
    ),
    "spinnaker2": ChipSpec(
        name="spinnaker2",
        vendor="University of Manchester / Dresden",
        total_cores=152,
        core=CoreSpec(
            max_neurons=1024,
            max_synapses_per_neuron=16384,
            weight_bits=16,
            supported_neuron_types=["LIF", "Izhikevich", "HH", "Custom"],
            has_on_chip_learning=True,
            learning_rules=["STDP", "R-STDP", "custom"],
            max_delay_steps=256,
        ),
        clock_mhz=500,
        power_mw_per_core=2.0,
        routing_topology="mesh",
        max_fan_out=16384,
        analog_noise_cv=0.0,
    ),
    "brainscales2": ChipSpec(
        name="brainscales2",
        vendor="University of Heidelberg",
        total_cores=1,
        core=CoreSpec(
            max_neurons=512,
            max_synapses_per_neuron=256,
            weight_bits=6,
            supported_neuron_types=["AdEx", "LIF"],
            has_on_chip_learning=True,
            learning_rules=["STDP", "correlation"],
            max_delay_steps=4,
        ),
        clock_mhz=125,
        power_mw_per_core=30.0,
        routing_topology="crossbar",
        max_fan_out=256,
        analog_noise_cv=0.20,  # analog mixed-signal: ~20% CV
    ),
}
