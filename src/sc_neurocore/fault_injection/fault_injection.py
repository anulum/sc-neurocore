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
from typing import List, Tuple

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
        self.rng = np.random.default_rng(seed)

    def inject(
        self,
        bitstream: np.ndarray,
        model: FaultModel,
        ber: float,
    ) -> Tuple[np.ndarray, int]:
        """Inject faults into a boolean bitstream.

        Returns (corrupted_bitstream, num_bits_affected).
        """
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

        return corrupted, 0

    def inject_at_positions(
        self,
        bitstream: np.ndarray,
        positions: List[int],
    ) -> np.ndarray:
        """Flip specific bit positions (deterministic injection)."""
        corrupted = bitstream.copy()
        for pos in positions:
            if 0 <= pos < len(corrupted):
                corrupted[pos] = 1 - corrupted[pos]
        return corrupted


class ResilienceBenchmark:
    """Monte-Carlo resilience benchmarking harness."""

    def __init__(self, seed: int = 42):
        self.injector = FaultInjector(seed=seed)
        self.rng = np.random.default_rng(seed)

    def _generate_bitstream(self, length: int, probability: float) -> np.ndarray:
        """Generate a random SC bitstream encoding a given probability."""
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
