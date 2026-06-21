# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
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

Specs can be loaded from JSON for custom/future chips.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

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
    """Load and validate a chip spec from a JSON file."""
    source = Path(path)
    try:
        with source.open(encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source} is not valid chip spec JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("chip spec JSON root must be an object")

    chip = _validate_chip_payload(payload, source=str(source))
    core = _validate_core_payload(payload["core"], source=str(source))
    return ChipSpec(core=core, **chip)


_REQUIRED_CHIP_FIELDS = ("name", "vendor", "total_cores", "core")
_OPTIONAL_CHIP_FIELDS = (
    "clock_mhz",
    "power_mw_per_core",
    "routing_topology",
    "max_fan_out",
    "analog_noise_cv",
)
_REQUIRED_CORE_FIELDS = (
    "max_neurons",
    "max_synapses_per_neuron",
    "weight_bits",
    "supported_neuron_types",
)
_OPTIONAL_CORE_FIELDS = ("has_on_chip_learning", "learning_rules", "max_delay_steps")
_ROUTING_TOPOLOGIES = {"mesh", "crossbar", "tree", "ring"}


def _validate_chip_payload(payload: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    _validate_key_set(
        payload,
        required=_REQUIRED_CHIP_FIELDS,
        optional=_OPTIONAL_CHIP_FIELDS,
        source=source,
        label="chip spec",
    )
    chip: dict[str, Any] = {
        "name": _required_str(payload, "name", source),
        "vendor": _required_str(payload, "vendor", source),
        "total_cores": _required_positive_int(payload, "total_cores", source),
    }

    if "clock_mhz" in payload:
        chip["clock_mhz"] = _required_positive_float(payload, "clock_mhz", source)
    if "power_mw_per_core" in payload:
        chip["power_mw_per_core"] = _required_non_negative_float(
            payload, "power_mw_per_core", source
        )
    if "routing_topology" in payload:
        routing = _required_str(payload, "routing_topology", source)
        if routing not in _ROUTING_TOPOLOGIES:
            raise ValueError(
                f"{source}: routing_topology must be one of {sorted(_ROUTING_TOPOLOGIES)}"
            )
        chip["routing_topology"] = routing
    if "max_fan_out" in payload:
        chip["max_fan_out"] = _required_non_negative_int(payload, "max_fan_out", source)
    if "analog_noise_cv" in payload:
        chip["analog_noise_cv"] = _required_non_negative_float(payload, "analog_noise_cv", source)
    return chip


def _validate_core_payload(value: Any, *, source: str) -> CoreSpec:
    if not isinstance(value, dict):
        raise ValueError(f"{source}: core must be an object")
    _validate_key_set(
        value,
        required=_REQUIRED_CORE_FIELDS,
        optional=_OPTIONAL_CORE_FIELDS,
        source=source,
        label="core spec",
    )
    core: dict[str, Any] = {
        "max_neurons": _required_positive_int(value, "max_neurons", source),
        "max_synapses_per_neuron": _required_positive_int(value, "max_synapses_per_neuron", source),
        "weight_bits": _required_positive_int(value, "weight_bits", source),
        "supported_neuron_types": _required_str_list(value, "supported_neuron_types", source),
    }
    if "has_on_chip_learning" in value:
        core["has_on_chip_learning"] = _required_bool(value, "has_on_chip_learning", source)
    if "learning_rules" in value:
        core["learning_rules"] = _required_str_list(value, "learning_rules", source)
    if "max_delay_steps" in value:
        core["max_delay_steps"] = _required_non_negative_int(value, "max_delay_steps", source)
    return CoreSpec(**core)


def _validate_key_set(
    payload: Mapping[str, Any],
    *,
    required: tuple[str, ...],
    optional: tuple[str, ...],
    source: str,
    label: str,
) -> None:
    keys = set(payload)
    missing = sorted(set(required) - keys)
    extra = sorted(keys - set(required) - set(optional))
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"unexpected={extra}")
        raise ValueError(f"{source}: invalid {label} fields: {', '.join(details)}")


def _required_str(payload: Mapping[str, Any], key: str, source: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source}: {key} must be a non-empty string")
    return value


def _required_bool(payload: Mapping[str, Any], key: str, source: str) -> bool:
    value = payload[key]
    if not isinstance(value, bool):
        raise ValueError(f"{source}: {key} must be boolean")
    return value


def _required_str_list(payload: Mapping[str, Any], key: str, source: str) -> list[str]:
    value = payload[key]
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"{source}: {key} must be a non-empty list of strings")
    return list(value)


def _required_positive_int(payload: Mapping[str, Any], key: str, source: str) -> int:
    value = _required_int(payload, key, source)
    if value <= 0:
        raise ValueError(f"{source}: {key} must be positive")
    return value


def _required_non_negative_int(payload: Mapping[str, Any], key: str, source: str) -> int:
    value = _required_int(payload, key, source)
    if value < 0:
        raise ValueError(f"{source}: {key} must be non-negative")
    return value


def _required_int(payload: Mapping[str, Any], key: str, source: str) -> int:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{source}: {key} must be an integer")
    return cast(int, value)


def _required_positive_float(payload: Mapping[str, Any], key: str, source: str) -> float:
    value = _required_float(payload, key, source)
    if value <= 0.0:
        raise ValueError(f"{source}: {key} must be positive")
    return value


def _required_non_negative_float(payload: Mapping[str, Any], key: str, source: str) -> float:
    value = _required_float(payload, key, source)
    if value < 0.0:
        raise ValueError(f"{source}: {key} must be non-negative")
    return value


def _required_float(payload: Mapping[str, Any], key: str, source: str) -> float:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{source}: {key} must be numeric")
    result = float(value)
    if not result == result or result in {float("inf"), float("-inf")}:
        raise ValueError(f"{source}: {key} must be finite")
    return result


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
