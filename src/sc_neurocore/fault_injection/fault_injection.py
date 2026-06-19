# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fault Injection & Resilience Benchmarking

"""Monte-Carlo fault injection and resilience benchmarking for SC bitstreams.

Supports bit-flip, stuck-at, Gaussian noise, and dropout fault models
at configurable bit error rates. Includes radiation environment presets
for LEO, GEO, and deep-space applications.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Tuple

import numpy as np


class FaultModel(Enum):
    BIT_FLIP = "bit_flip"
    STUCK_AT_0 = "stuck_at_0"
    STUCK_AT_1 = "stuck_at_1"
    GAUSSIAN_NOISE = "gaussian_noise"
    DROPOUT = "dropout"


@dataclass
class RadiationProfile:
    """Preset radiation environments with typical BER (bit error rate)."""

    name: str
    ber: float  # bit error rate per bit per cycle
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if isinstance(self.ber, bool) or not isinstance(self.ber, int | float):
            raise ValueError("ber must be a finite value in [0, 1]")
        if not np.isfinite(float(self.ber)) or float(self.ber) < 0.0 or float(self.ber) > 1.0:
            raise ValueError("ber must be a finite value in [0, 1]")
        if not isinstance(self.description, str):
            raise ValueError("description must be a string")

    @classmethod
    def leo(cls) -> RadiationProfile:
        return cls("LEO", 1e-7, "Low Earth Orbit — moderate radiation belt exposure")

    @classmethod
    def geo(cls) -> RadiationProfile:
        return cls("GEO", 5e-6, "Geostationary — prolonged Van Allen belt exposure")

    @classmethod
    def deep_space(cls) -> RadiationProfile:
        return cls("Deep Space", 1e-4, "Interplanetary — galactic cosmic rays")

    @classmethod
    def terrestrial(cls) -> RadiationProfile:
        return cls("Terrestrial", 1e-10, "Sea-level — thermal neutron background")


@dataclass
class FaultInjectionResult:
    original_popcount: int
    corrupted_popcount: int
    bits_flipped: int
    bitstream_length: int

    def __post_init__(self) -> None:
        for field_name in (
            "original_popcount",
            "corrupted_popcount",
            "bits_flipped",
            "bitstream_length",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field_name} must be an integer")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.original_popcount > self.bitstream_length:
            raise ValueError("original_popcount cannot exceed bitstream_length")
        if self.corrupted_popcount > self.bitstream_length:
            raise ValueError("corrupted_popcount cannot exceed bitstream_length")
        if self.bits_flipped > self.bitstream_length:
            raise ValueError("bits_flipped cannot exceed bitstream_length")

    @property
    def probability_original(self) -> float:
        return self.original_popcount / self.bitstream_length if self.bitstream_length > 0 else 0.0

    @property
    def probability_corrupted(self) -> float:
        return self.corrupted_popcount / self.bitstream_length if self.bitstream_length > 0 else 0.0

    @property
    def absolute_error(self) -> float:
        return abs(self.probability_original - self.probability_corrupted)


@dataclass
class ResilienceReport:
    fault_model: str
    ber: float
    bitstream_length: int
    num_trials: int
    mean_error: float
    std_error: float
    max_error: float
    p95_error: float
    p99_error: float
    mean_bits_flipped: float
    wall_time_ms: float

    def __post_init__(self) -> None:
        if not isinstance(self.fault_model, str) or not self.fault_model.strip():
            raise ValueError("fault_model must be a non-empty string")
        valid_fault_models = {member.value for member in FaultModel}
        if self.fault_model not in valid_fault_models:
            raise ValueError("fault_model must reference a known FaultModel value")
        numeric_fields = (
            "ber",
            "mean_error",
            "std_error",
            "max_error",
            "p95_error",
            "p99_error",
            "mean_bits_flipped",
            "wall_time_ms",
        )
        for field_name in numeric_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{field_name} must be numeric")
            if not np.isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite")
        for field_name in ("bitstream_length", "num_trials"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.ber < 0.0 or self.ber > 1.0:
            raise ValueError("ber must be in [0, 1]")
        for field_name in ("mean_error", "std_error", "max_error", "p95_error", "p99_error"):
            if getattr(self, field_name) < 0.0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.mean_bits_flipped < 0.0 or self.mean_bits_flipped > self.bitstream_length:
            raise ValueError("mean_bits_flipped must be in [0, bitstream_length]")
        if self.wall_time_ms < 0.0:
            raise ValueError("wall_time_ms must be non-negative")
        if self.p95_error < self.mean_error:
            raise ValueError("p95_error must be >= mean_error")
        if self.p99_error < self.p95_error:
            raise ValueError("p99_error must be >= p95_error")
        if self.max_error < self.p99_error:
            raise ValueError("max_error must be >= p99_error")

    def summary(self) -> str:
        return (
            f"Fault: {self.fault_model}, BER: {self.ber:.2e}, "
            f"N={self.bitstream_length}, Trials={self.num_trials}\n"
            f"  Mean Error: {self.mean_error:.6f} ± {self.std_error:.6f}\n"
            f"  P95: {self.p95_error:.6f}, P99: {self.p99_error:.6f}, "
            f"Max: {self.max_error:.6f}\n"
            f"  Mean Bits Flipped: {self.mean_bits_flipped:.2f}\n"
            f"  Wall Time: {self.wall_time_ms:.2f} ms"
        )


class FaultInjector:
    """Applies configurable faults to SC bitstreams."""

    def __init__(self, seed: int = 42):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self.rng = np.random.default_rng(seed)

    def inject(
        self,
        bitstream: np.ndarray[Any, Any],
        model: FaultModel,
        ber: float,
    ) -> Tuple[np.ndarray[Any, Any], int]:
        """Inject faults into a boolean bitstream.

        Returns (corrupted_bitstream, num_bits_affected).
        """
        if not isinstance(bitstream, np.ndarray):
            raise ValueError("bitstream must be a numpy.ndarray")
        if bitstream.ndim != 1:
            raise ValueError("bitstream must be a 1-D array")
        if bitstream.size == 0:
            raise ValueError("bitstream must be non-empty")
        if not isinstance(model, FaultModel):
            raise ValueError("model must be a FaultModel")
        if isinstance(ber, bool) or not isinstance(ber, int | float):
            raise ValueError("ber must be a finite value in [0, 1]")
        if not np.isfinite(float(ber)) or float(ber) < 0.0 or float(ber) > 1.0:
            raise ValueError("ber must be a finite value in [0, 1]")
        if model in (
            FaultModel.BIT_FLIP,
            FaultModel.STUCK_AT_0,
            FaultModel.STUCK_AT_1,
            FaultModel.DROPOUT,
        ):
            unique_values = np.unique(bitstream)
            if not np.isin(unique_values, np.array([0, 1])).all():
                raise ValueError("discrete fault models require binary bitstreams")
        if model == FaultModel.GAUSSIAN_NOISE:
            if not np.issubdtype(bitstream.dtype, np.number):
                raise ValueError("gaussian_noise requires numeric bitstreams")
            values = bitstream.astype(np.float64)
            if not np.isfinite(values).all():
                raise ValueError("gaussian_noise requires finite bitstream values")
            if (values < 0.0).any() or (values > 1.0).any():
                raise ValueError("gaussian_noise requires values within [0, 1]")
        if ber == 0.0:
            return bitstream.copy(), 0

        corrupted = bitstream.copy()
        n = len(bitstream)

        if model == FaultModel.BIT_FLIP:
            mask = self.rng.random(n) < ber
            corrupted = np.logical_xor(bitstream, mask)
            return corrupted.astype(bitstream.dtype), int(np.sum(mask))

        if model == FaultModel.STUCK_AT_0:
            mask = self.rng.random(n) < ber
            corrupted[mask] = 0
            affected = int(np.sum(mask & bitstream.astype(bool)))
            return corrupted, affected

        if model == FaultModel.STUCK_AT_1:
            mask = self.rng.random(n) < ber
            corrupted[mask] = 1
            affected = int(np.sum(mask & ~bitstream.astype(bool)))
            return corrupted, affected

        if model == FaultModel.GAUSSIAN_NOISE:
            # For continuous-valued SC streams (e.g., analog approximation)
            noise = self.rng.normal(0, ber, n)
            corrupted = np.clip(bitstream.astype(np.float64) + noise, 0, 1)
            corrupted = (corrupted > 0.5).astype(bitstream.dtype)
            changed = int(np.sum(corrupted != bitstream))
            return corrupted, changed

        if model == FaultModel.DROPOUT:
            mask = self.rng.random(n) < ber
            corrupted[mask] = 0
            affected = int(np.sum(mask & bitstream.astype(bool)))
            return corrupted, affected

        raise ValueError(f"unsupported fault model: {model}")

    def inject_at_positions(
        self,
        bitstream: np.ndarray[Any, Any],
        positions: List[int],
    ) -> np.ndarray[Any, Any]:
        """Flip specific bit positions (deterministic injection)."""
        if not isinstance(bitstream, np.ndarray):
            raise ValueError("bitstream must be a numpy.ndarray")
        if bitstream.ndim != 1:
            raise ValueError("bitstream must be a 1-D array")
        if not isinstance(positions, list):
            raise ValueError("positions must be a list of integers")
        seen_positions = set()
        for pos in positions:
            if isinstance(pos, bool) or not isinstance(pos, int):
                raise ValueError("positions must contain integers")
            if pos in seen_positions:
                raise ValueError("positions must be unique")
            seen_positions.add(pos)
            if pos < 0 or pos >= len(bitstream):
                raise ValueError("positions must be within bitstream bounds")

        corrupted = bitstream.copy()
        for pos in positions:
            corrupted[pos] = 1 - corrupted[pos]
        return corrupted


class ResilienceBenchmark:
    """Monte-Carlo resilience benchmarking harness."""

    def __init__(self, seed: int = 42):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self.injector = FaultInjector(seed=seed)
        self.rng = np.random.default_rng(seed)

    def _generate_bitstream(self, length: int, probability: float) -> np.ndarray[Any, Any]:
        """Generate a random SC bitstream encoding a given probability."""
        if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
            raise ValueError("length must be a positive integer")
        if isinstance(probability, bool) or not isinstance(probability, int | float):
            raise ValueError("probability must be a finite value in [0, 1]")
        if (
            not np.isfinite(float(probability))
            or float(probability) < 0.0
            or float(probability) > 1.0
        ):
            raise ValueError("probability must be a finite value in [0, 1]")
        return (self.rng.random(length) < probability).astype(np.uint8)

    def run(
        self,
        *,
        fault_model: FaultModel,
        ber: float,
        bitstream_length: int = 1024,
        probability: float = 0.5,
        num_trials: int = 1000,
    ) -> ResilienceReport:
        """Run Monte-Carlo fault injection trials."""
        if not isinstance(fault_model, FaultModel):
            raise ValueError("fault_model must be a FaultModel")
        if (
            isinstance(bitstream_length, bool)
            or not isinstance(bitstream_length, int)
            or bitstream_length <= 0
        ):
            raise ValueError("bitstream_length must be a positive integer")
        if isinstance(num_trials, bool) or not isinstance(num_trials, int) or num_trials <= 0:
            raise ValueError("num_trials must be a positive integer")
        if isinstance(probability, bool) or not isinstance(probability, int | float):
            raise ValueError("probability must be a finite value in [0, 1]")
        if (
            not np.isfinite(float(probability))
            or float(probability) < 0.0
            or float(probability) > 1.0
        ):
            raise ValueError("probability must be a finite value in [0, 1]")
        if isinstance(ber, bool) or not isinstance(ber, int | float):
            raise ValueError("ber must be a finite value in [0, 1]")
        if not np.isfinite(float(ber)) or float(ber) < 0.0 or float(ber) > 1.0:
            raise ValueError("ber must be a finite value in [0, 1]")

        errors = []
        bits_flipped_list = []
        start = time.perf_counter()

        for _ in range(num_trials):
            bs = self._generate_bitstream(bitstream_length, probability)
            original_pc = int(np.sum(bs))

            corrupted, n_flipped = self.injector.inject(bs, fault_model, ber)
            corrupted_pc = int(np.sum(corrupted))

            result = FaultInjectionResult(
                original_popcount=original_pc,
                corrupted_popcount=corrupted_pc,
                bits_flipped=n_flipped,
                bitstream_length=bitstream_length,
            )
            errors.append(result.absolute_error)
            bits_flipped_list.append(n_flipped)

        wall_time = (time.perf_counter() - start) * 1000.0
        errors_arr = np.array(errors)
        flipped_arr = np.array(bits_flipped_list, dtype=np.float64)
        if errors_arr.shape != (num_trials,):
            raise ValueError("internal error: error vector shape mismatch")
        if flipped_arr.shape != (num_trials,):
            raise ValueError("internal error: flipped vector shape mismatch")
        if not np.isfinite(errors_arr).all():
            raise ValueError("internal error: non-finite error values produced")
        if not np.isfinite(flipped_arr).all():
            raise ValueError("internal error: non-finite flipped-count values produced")
        if (errors_arr < 0.0).any():
            raise ValueError("internal error: negative error values produced")
        if (flipped_arr < 0.0).any() or (flipped_arr > bitstream_length).any():
            raise ValueError("internal error: flipped-count values out of range")

        return ResilienceReport(
            fault_model=fault_model.value,
            ber=ber,
            bitstream_length=bitstream_length,
            num_trials=num_trials,
            mean_error=float(np.mean(errors_arr)),
            std_error=float(np.std(errors_arr)),
            max_error=float(np.max(errors_arr)),
            p95_error=float(np.percentile(errors_arr, 95)),
            p99_error=float(np.percentile(errors_arr, 99)),
            mean_bits_flipped=float(np.mean(flipped_arr)),
            wall_time_ms=wall_time,
        )

    def sweep_ber(
        self,
        *,
        fault_model: FaultModel,
        ber_range: List[float],
        bitstream_length: int = 1024,
        probability: float = 0.5,
        num_trials: int = 500,
    ) -> List[ResilienceReport]:
        """Sweep across multiple BER values to produce a degradation curve."""
        if not isinstance(fault_model, FaultModel):
            raise ValueError("fault_model must be a FaultModel")
        if not isinstance(ber_range, list) or not ber_range:
            raise ValueError("ber_range must be a non-empty list")
        for ber in ber_range:
            if isinstance(ber, bool) or not isinstance(ber, int | float):
                raise ValueError("ber_range entries must be finite values in [0, 1]")
            if not np.isfinite(float(ber)) or float(ber) < 0.0 or float(ber) > 1.0:
                raise ValueError("ber_range entries must be finite values in [0, 1]")
        if any(ber_range[i] > ber_range[i + 1] for i in range(len(ber_range) - 1)):
            raise ValueError("ber_range must be monotonically non-decreasing")

        return [
            self.run(
                fault_model=fault_model,
                ber=ber,
                bitstream_length=bitstream_length,
                probability=probability,
                num_trials=num_trials,
            )
            for ber in ber_range
        ]


if __name__ == "__main__":
    bench = ResilienceBenchmark(seed=0)

    print("=== SC Resilience Benchmark ===\n")
    for profile in [
        RadiationProfile.terrestrial(),
        RadiationProfile.leo(),
        RadiationProfile.geo(),
        RadiationProfile.deep_space(),
    ]:
        report = bench.run(
            fault_model=FaultModel.BIT_FLIP,
            ber=profile.ber,
            bitstream_length=1024,
            num_trials=1000,
        )
        print(f"[{profile.name}] {profile.description}")
        print(report.summary())
        print()
