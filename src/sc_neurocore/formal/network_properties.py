# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Network-level formal property contracts

from __future__ import annotations

from dataclasses import dataclass
import re

_SV_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


def validate_systemverilog_identifier(value: str, *, field_name: str) -> str:
    """Return ``value`` after checking it is a plain SystemVerilog identifier."""
    if not isinstance(value, str) or not _SV_IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a valid SystemVerilog identifier")
    return value


def _validate_positive_int(value: int, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _validate_non_negative_int(value: int, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class DenseLIFNetworkSpec:
    """Formal boundary contract for a dense LIF network HDL module."""

    name: str
    input_width: int
    output_width: int
    state_width: int
    timestep_name: str = "sample_valid"
    output_signal: str = "spike_out"
    clock_name: str = "clk"
    reset_name: str = "rst_n"

    def __post_init__(self) -> None:
        validate_systemverilog_identifier(self.name, field_name="name")
        validate_systemverilog_identifier(self.timestep_name, field_name="timestep_name")
        validate_systemverilog_identifier(self.output_signal, field_name="output_signal")
        validate_systemverilog_identifier(self.clock_name, field_name="clock_name")
        validate_systemverilog_identifier(self.reset_name, field_name="reset_name")
        _validate_positive_int(self.input_width, field_name="input_width")
        _validate_positive_int(self.output_width, field_name="output_width")
        _validate_positive_int(self.state_width, field_name="state_width")


@dataclass(frozen=True, slots=True)
class NetworkRateBound:
    """Bound a selected output neuron's spike count inside a fixed time window."""

    name: str
    output_index: int
    window_cycles: int
    max_spikes: int

    def __post_init__(self) -> None:
        validate_systemverilog_identifier(self.name, field_name="name")
        _validate_non_negative_int(self.output_index, field_name="output_index")
        _validate_positive_int(self.window_cycles, field_name="window_cycles")
        _validate_non_negative_int(self.max_spikes, field_name="max_spikes")
        if self.max_spikes > self.window_cycles:
            raise ValueError("max_spikes must be less than or equal to window_cycles")


@dataclass(frozen=True, slots=True)
class NetworkRefractoryInvariant:
    """Forbid a selected output from spiking during its refractory window."""

    name: str
    output_index: int
    refractory_cycles: int

    def __post_init__(self) -> None:
        validate_systemverilog_identifier(self.name, field_name="name")
        _validate_non_negative_int(self.output_index, field_name="output_index")
        _validate_positive_int(self.refractory_cycles, field_name="refractory_cycles")


@dataclass(frozen=True, slots=True)
class NetworkAntagonisticOutputExclusion:
    """Forbid two antagonistic network outputs from spiking in the same cycle."""

    name: str
    output_a: int
    output_b: int

    def __post_init__(self) -> None:
        validate_systemverilog_identifier(self.name, field_name="name")
        _validate_non_negative_int(self.output_a, field_name="output_a")
        _validate_non_negative_int(self.output_b, field_name="output_b")
        if self.output_a == self.output_b:
            raise ValueError("output_a and output_b must be distinct")


@dataclass(frozen=True, slots=True)
class NetworkOutputTemporalSeparation:
    """Forbid two outputs from spiking within a bounded cycle window."""

    name: str
    output_a: int
    output_b: int
    separation_cycles: int

    def __post_init__(self) -> None:
        validate_systemverilog_identifier(self.name, field_name="name")
        _validate_non_negative_int(self.output_a, field_name="output_a")
        _validate_non_negative_int(self.output_b, field_name="output_b")
        if self.output_a == self.output_b:
            raise ValueError("output_a and output_b must be distinct")
        _validate_positive_int(self.separation_cycles, field_name="separation_cycles")


@dataclass(frozen=True, slots=True)
class NetworkPopulationCoactivationCap:
    """Bound the number of simultaneously active outputs in a sample cycle."""

    name: str
    max_active_outputs: int

    def __post_init__(self) -> None:
        validate_systemverilog_identifier(self.name, field_name="name")
        _validate_non_negative_int(self.max_active_outputs, field_name="max_active_outputs")


@dataclass(frozen=True, slots=True)
class NetworkPopulationSilenceAfterCoactivation:
    """Require global output silence after a population coactivation event."""

    name: str
    trigger_active_outputs: int
    silence_cycles: int

    def __post_init__(self) -> None:
        validate_systemverilog_identifier(self.name, field_name="name")
        _validate_positive_int(
            self.trigger_active_outputs,
            field_name="trigger_active_outputs",
        )
        _validate_positive_int(self.silence_cycles, field_name="silence_cycles")


@dataclass(frozen=True, slots=True)
class NetworkPopulationInactivityBound:
    """Bound consecutive valid cycles with no active network outputs."""

    name: str
    max_silent_cycles: int

    def __post_init__(self) -> None:
        validate_systemverilog_identifier(self.name, field_name="name")
        _validate_positive_int(self.max_silent_cycles, field_name="max_silent_cycles")
