# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Annealing schedules, gauges, and SC precision encodings

"""Deterministic schedule, gauge, and probability transformations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from sc_neurocore.bridges.annealing_models import IsingModel


def _finite(name: str, value: float) -> float:
    """Return a finite float."""
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _positive_duration(name: str, value: float) -> float:
    """Return a finite positive duration."""
    duration = _finite(name, value)
    if duration <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return duration


def _anneal_fraction(name: str, value: float) -> float:
    """Return an anneal fraction in the closed unit interval."""
    fraction = _finite(name, value)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return fraction


class AnnealingSchedule:
    """Build validated linear, pause, and reverse annealing schedules."""

    def __init__(self) -> None:
        """Create an empty schedule."""
        self._points: list[tuple[float, float]] = []

    def linear(self, duration_us: float = 20.0) -> AnnealingSchedule:
        """Configure a standard linear anneal from zero to one."""
        duration = _positive_duration("duration_us", duration_us)
        self._points = [(0.0, 0.0), (duration, 1.0)]
        return self

    def pause_and_quench(
        self,
        ramp_time_us: float = 5.0,
        pause_at_s: float = 0.4,
        pause_duration_us: float = 50.0,
        quench_time_us: float = 1.0,
    ) -> AnnealingSchedule:
        """Ramp, hold at an intermediate fraction, then quench."""
        ramp = _positive_duration("ramp_time_us", ramp_time_us)
        pause = _positive_duration("pause_duration_us", pause_duration_us)
        quench = _positive_duration("quench_time_us", quench_time_us)
        fraction = _anneal_fraction("pause_at_s", pause_at_s)
        if fraction in {0.0, 1.0}:
            raise ValueError("pause_at_s must be strictly between zero and one")
        self._points = [
            (0.0, 0.0),
            (ramp, fraction),
            (ramp + pause, fraction),
            (ramp + pause + quench, 1.0),
        ]
        return self

    def reverse(
        self,
        initial_s: float = 1.0,
        reverse_to_s: float = 0.3,
        ramp_time_us: float = 5.0,
        hold_time_us: float = 10.0,
        forward_time_us: float = 5.0,
    ) -> AnnealingSchedule:
        """Configure a reverse anneal followed by a forward return."""
        initial = _anneal_fraction("initial_s", initial_s)
        reverse_to = _anneal_fraction("reverse_to_s", reverse_to_s)
        if reverse_to >= initial:
            raise ValueError("reverse_to_s must be smaller than initial_s")
        ramp = _positive_duration("ramp_time_us", ramp_time_us)
        hold = _positive_duration("hold_time_us", hold_time_us)
        forward = _positive_duration("forward_time_us", forward_time_us)
        self._points = [
            (0.0, initial),
            (ramp, reverse_to),
            (ramp + hold, reverse_to),
            (ramp + hold + forward, 1.0),
        ]
        return self

    @property
    def points(self) -> list[tuple[float, float]]:
        """Return a defensive copy of the schedule points."""
        return list(self._points)

    @property
    def total_time_us(self) -> float:
        """Return zero for an empty schedule or its final timestamp."""
        return self._points[-1][0] if self._points else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a D-Wave-compatible schedule payload."""
        return {
            "schedule": list(self._points),
            "total_time_us": self.total_time_us,
            "n_points": len(self._points),
        }


class GaugeTransform:
    """Generate deterministic random spin-reversal transformations."""

    def __init__(self, n_gauges: int = 10, seed: int = 42) -> None:
        """Configure a positive transform count and deterministic seed."""
        if isinstance(n_gauges, bool) or not isinstance(n_gauges, int) or n_gauges <= 0:
            raise ValueError("n_gauges must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self._n_gauges = n_gauges
        self._rng = np.random.default_rng(seed)

    def transform(self, model: IsingModel) -> list[IsingModel]:
        """Return energy-equivalent gauge-transformed model copies."""
        if not isinstance(model, IsingModel) or model.n_qubits <= 0:
            raise ValueError("model must be a non-empty IsingModel")
        transforms: list[IsingModel] = []
        for gauge_index in range(self._n_gauges):
            gauge = {index: int(self._rng.choice((-1, 1))) for index in range(model.n_qubits)}
            transforms.append(
                IsingModel(
                    h={index: gauge[index] * bias for index, bias in model.h.items()},
                    J={
                        pair: gauge[pair[0]] * gauge[pair[1]] * strength
                        for pair, strength in model.J.items()
                    },
                    offset=model.offset,
                    qubit_labels=dict(model.qubit_labels),
                    n_qubits=model.n_qubits,
                    source=f"{model.source}_gauge{gauge_index}",
                )
            )
        return transforms

    def untransform_sample(
        self,
        sample: Mapping[int, int],
        gauge: Mapping[int, int],
    ) -> dict[int, int]:
        """Return a transformed sample to the original spin frame."""
        for mapping_name, mapping in (("sample", sample), ("gauge", gauge)):
            for index, spin in mapping.items():
                if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                    raise ValueError(f"{mapping_name} indices must be non-negative integers")
                if spin not in {-1, 1}:
                    raise ValueError(f"{mapping_name} values must be -1 or +1")
        return {index: spin * gauge.get(index, 1) for index, spin in sample.items()}


class SCPrecisionEncoder:
    """Encode unit-interval SC values as binary, unary, or one-hot qubits."""

    def __init__(self, encoding: str = "binary", n_bits: int = 8) -> None:
        """Select a supported encoding and positive qubit count."""
        if encoding not in {"binary", "unary", "one_hot"}:
            raise ValueError(f"Unknown encoding: {encoding}")
        if isinstance(n_bits, bool) or not isinstance(n_bits, int) or n_bits <= 0:
            raise ValueError("n_bits must be a positive integer")
        self._encoding = encoding
        self._n_bits = n_bits

    @property
    def n_levels(self) -> int:
        """Return the number of representable precision levels."""
        if self._encoding == "binary":
            return int(2**self._n_bits)
        if self._encoding == "unary":
            return self._n_bits + 1
        return self._n_bits

    def encode(self, sc_value: float) -> dict[int, int]:
        """Encode one finite SC value after clipping it to ``[0, 1]``."""
        value = _finite("sc_value", sc_value)
        clipped = max(0.0, min(1.0, value))
        if self._encoding == "binary":
            level = int(round(clipped * (2**self._n_bits - 1)))
            return {index: (level >> index) & 1 for index in range(self._n_bits)}
        if self._encoding == "unary":
            one_count = int(round(clipped * self._n_bits))
            return {index: 1 if index < one_count else 0 for index in range(self._n_bits)}
        level = int(round(clipped * (self._n_bits - 1)))
        return {index: 1 if index == level else 0 for index in range(self._n_bits)}

    def decode(self, qubits: Mapping[int, int]) -> float:
        """Decode a validated partial binary qubit mapping."""
        for index, bit in qubits.items():
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < self._n_bits
            ):
                raise ValueError("qubit indices must fit within the configured encoding")
            if bit not in {0, 1}:
                raise ValueError("qubit values must be binary")
        if self._encoding == "binary":
            level = sum(qubits.get(index, 0) << index for index in range(self._n_bits))
            return float(level / (2**self._n_bits - 1))
        if self._encoding == "unary":
            return sum(qubits.get(index, 0) for index in range(self._n_bits)) / self._n_bits
        active = [index for index in range(self._n_bits) if qubits.get(index, 0) == 1]
        if len(active) > 1:
            raise ValueError("one_hot decoding accepts at most one active qubit")
        return active[0] / max(self._n_bits - 1, 1) if active else 0.0

    def qubits_needed(self, n_sc_values: int) -> int:
        """Return the total qubits required for a non-negative value count."""
        if isinstance(n_sc_values, bool) or not isinstance(n_sc_values, int) or n_sc_values < 0:
            raise ValueError("n_sc_values must be a non-negative integer")
        return n_sc_values * self._n_bits

    def encode_array(self, values: np.ndarray[Any, Any]) -> dict[int, int]:
        """Encode a non-empty one-dimensional array into global qubit indices."""
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1 or array.size == 0:
            raise ValueError("values must be a non-empty one-dimensional array")
        if not bool(np.all(np.isfinite(array))):
            raise ValueError("values must contain only finite numbers")
        result: dict[int, int] = {}
        for value_index, value in enumerate(array):
            for local_index, bit in self.encode(float(value)).items():
                result[value_index * self._n_bits + local_index] = bit
        return result
