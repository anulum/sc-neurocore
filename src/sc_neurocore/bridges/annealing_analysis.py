# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quantum-annealing analysis and metrics

"""Energy-landscape, embedding, sampling, and TTS analysis."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from sc_neurocore.bridges import annealing_backends as backends
from sc_neurocore.bridges.annealing_models import (
    BackendChoice,
    IsingModel,
    validate_backend_choice,
)


def _finite(name: str, value: object) -> float:
    """Return a finite numeric value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _validated_sample(sample: Mapping[int, int], *, spins_only: bool = True) -> dict[int, int]:
    """Validate and copy a spin or binary sample."""
    allowed = {-1, 1} if spins_only else {-1, 0, 1}
    normalized: dict[int, int] = {}
    for index, value in sample.items():
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("sample indices must be non-negative integers")
        if value not in allowed:
            raise ValueError("sample values are outside the supported domain")
        normalized[index] = value
    return normalized


class EnergyLandscape:
    """Compute energy statistics and spectral gaps for an Ising model."""

    def __init__(
        self,
        *,
        backend: BackendChoice = "auto",
        random_sample_count: int = 10_000,
        seed: int = 42,
    ) -> None:
        """Configure deterministic large-model sampling and backend choice."""
        self._backend = validate_backend_choice(backend)
        if (
            isinstance(random_sample_count, bool)
            or not isinstance(random_sample_count, int)
            or random_sample_count <= 0
        ):
            raise ValueError("random_sample_count must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self._random_sample_count = random_sample_count
        self._seed = seed

    def analyze(
        self,
        model: IsingModel,
        samples: Sequence[Mapping[int, int]] | None = None,
    ) -> dict[str, Any]:
        """Analyze exhaustive, supplied, or deterministic random samples."""
        if not isinstance(model, IsingModel) or model.n_qubits <= 0:
            raise ValueError("model must be a non-empty IsingModel")
        if samples is None:
            if model.n_qubits <= 20:
                normalized_samples = self._enumerate_all(model.n_qubits)
            else:
                rng = np.random.default_rng(self._seed)
                normalized_samples = [
                    {index: int(rng.choice((-1, 1))) for index in range(model.n_qubits)}
                    for _ in range(self._random_sample_count)
                ]
        else:
            if isinstance(samples, (str, bytes)):
                raise ValueError("samples must be a sequence of spin mappings")
            normalized_samples = [_validated_sample(sample) for sample in samples]
            if not normalized_samples:
                raise ValueError("samples must not be empty")

        use_rust = self._backend == "rust" or (
            self._backend == "auto" and backends.HAS_RUST_QA and len(normalized_samples) > 100
        )
        if use_rust:
            h_indices = list(model.h)
            j_pairs = list(model.J)
            raw_energies = backends.require_rust_batch_energy()(
                h_indices,
                [model.h[index] for index in h_indices],
                [pair[0] for pair in j_pairs],
                [pair[1] for pair in j_pairs],
                [model.J[pair] for pair in j_pairs],
                [
                    [sample.get(index, 1) for index in range(model.n_qubits)]
                    for sample in normalized_samples
                ],
                model.offset,
            )
            energies = [
                _finite(f"energies[{index}]", value) for index, value in enumerate(raw_energies)
            ]
            if len(energies) != len(normalized_samples):
                raise RuntimeError("native batch backend returned the wrong energy count")
        else:
            energies = [model.energy(sample, backend="python") for sample in normalized_samples]

        unique_energies = sorted(set(energies))
        minimum = unique_energies[0]
        spectral_gap = unique_energies[1] - unique_energies[0] if len(unique_energies) > 1 else 0.0
        return {
            "min_energy": minimum,
            "max_energy": max(energies),
            "mean_energy": float(np.mean(energies)),
            "std_energy": float(np.std(energies)),
            "spectral_gap": spectral_gap,
            "degeneracy": energies.count(minimum),
            "n_unique_energies": len(unique_energies),
            "n_samples": len(normalized_samples),
        }

    @staticmethod
    def _enumerate_all(n_qubits: int) -> list[dict[int, int]]:
        """Enumerate every configuration for at most 20 qubits."""
        if isinstance(n_qubits, bool) or not isinstance(n_qubits, int) or not 0 <= n_qubits <= 20:
            raise ValueError("n_qubits must be an integer between zero and 20")
        return [
            {index: 1 if (bits >> index) & 1 else -1 for index in range(n_qubits)}
            for bits in range(2**n_qubits)
        ]


class EmbeddingAnalyzer:
    """Estimate logical-to-physical embedding requirements."""

    def analyze(self, model: IsingModel) -> dict[str, Any]:
        """Return graph density, degree, and Pegasus chain estimates."""
        if not isinstance(model, IsingModel) or model.n_qubits <= 0:
            raise ValueError("model must be a non-empty IsingModel")
        size = model.n_qubits
        coupling_count = len(model.J)
        max_possible = size * (size - 1) // 2
        degree = {index: 0 for index in range(size)}
        for first, second in model.J:
            degree[first] += 1
            degree[second] += 1
        max_degree = max(degree.values())
        mean_degree = sum(degree.values()) / size
        chain_length = max(1, math.ceil(max_degree / 15))
        estimated_physical = size * chain_length
        return {
            "n_logical_qubits": size,
            "n_couplers": coupling_count,
            "density": coupling_count / max(max_possible, 1),
            "max_degree": max_degree,
            "mean_degree": float(mean_degree),
            "min_chain_estimate": chain_length,
            "estimated_physical_qubits": estimated_physical,
            "pegasus_compatible": estimated_physical <= 5000,
        }


class SampleAggregator:
    """Deduplicate samples and compute energy-distribution statistics."""

    def aggregate(
        self,
        samples: Sequence[Mapping[int, int]],
        energies: Sequence[float],
        temperature: float = 1.0,
    ) -> dict[str, Any]:
        """Aggregate an aligned sample and energy sequence."""
        if isinstance(samples, (str, bytes)) or isinstance(energies, (str, bytes)):
            raise ValueError("samples and energies must be sequences")
        if len(samples) != len(energies):
            raise ValueError("samples and energies must have equal lengths")
        if not samples:
            return {"unique_samples": 0, "best": {}, "histogram": {}}
        thermal = _finite("temperature", temperature)
        if thermal <= 0.0:
            raise ValueError("temperature must be greater than zero")

        normalized_samples = [_validated_sample(sample, spins_only=False) for sample in samples]
        normalized_energies = [
            _finite(f"energies[{index}]", value) for index, value in enumerate(energies)
        ]
        paired = sorted(zip(normalized_energies, normalized_samples), key=lambda item: item[0])
        best_energy, best_sample = paired[0]
        unique_samples = len({tuple(sorted(sample.items())) for _, sample in paired})

        energy_array = np.asarray(normalized_energies, dtype=np.float64)
        bin_count = max(min(20, len(set(normalized_energies))), 1)
        counts, bin_edges = np.histogram(energy_array, bins=bin_count)
        shifted_weights = np.exp(-(energy_array - min(normalized_energies)) / thermal)
        partition = float(np.sum(shifted_weights))
        ground_count = sum(1 for energy in normalized_energies if abs(energy - best_energy) < 1e-10)
        return {
            "unique_samples": unique_samples,
            "total_samples": len(normalized_samples),
            "best_sample": best_sample,
            "best_energy": best_energy,
            "mean_energy": float(np.mean(energy_array)),
            "std_energy": float(np.std(energy_array)),
            "boltzmann_avg_energy": float(np.sum(shifted_weights * energy_array) / partition),
            "success_probability": ground_count / len(normalized_energies),
            "gs_degeneracy": ground_count,
            "histogram": {
                "counts": counts.tolist(),
                "bin_edges": bin_edges.tolist(),
            },
        }


class TTSAnalyzer:
    """Compute time-to-solution from single-run success probability."""

    def compute(
        self,
        p_success: float,
        t_anneal_us: float,
        p_target: float = 0.99,
    ) -> dict[str, float]:
        """Compute the standard cumulative-success TTS metric."""
        success = _finite("p_success", p_success)
        target = _finite("p_target", p_target)
        anneal_time = _finite("t_anneal_us", t_anneal_us)
        if not 0.0 <= success <= 1.0:
            raise ValueError("p_success must be between zero and one")
        if not 0.0 < target < 1.0:
            raise ValueError("p_target must be strictly between zero and one")
        if anneal_time <= 0.0:
            raise ValueError("t_anneal_us must be greater than zero")
        if success == 0.0:
            return {
                "tts_us": float("inf"),
                "tts_ms": float("inf"),
                "n_runs_needed": float("inf"),
                "p_success": 0.0,
                "p_target": target,
            }
        if success == 1.0:
            return {
                "tts_us": anneal_time,
                "tts_ms": anneal_time / 1000.0,
                "n_runs_needed": 1.0,
                "p_success": 1.0,
                "p_target": target,
            }

        run_count = math.log1p(-target) / math.log1p(-success)
        tts = anneal_time * run_count
        return {
            "tts_us": tts,
            "tts_ms": tts / 1000.0,
            "n_runs_needed": run_count,
            "p_success": success,
            "p_target": target,
        }

    def from_samples(
        self,
        energies: Sequence[float],
        ground_state_energy: float,
        t_anneal_us: float = 20.0,
        tolerance: float = 1e-6,
        p_target: float = 0.99,
    ) -> dict[str, float]:
        """Estimate single-run success from observed energies."""
        if isinstance(energies, (str, bytes)):
            raise ValueError("energies must be a sequence")
        ground = _finite("ground_state_energy", ground_state_energy)
        threshold = _finite("tolerance", tolerance)
        if threshold <= 0.0:
            raise ValueError("tolerance must be greater than zero")
        normalized = [
            _finite(f"energies[{index}]", energy) for index, energy in enumerate(energies)
        ]
        ground_count = sum(1 for energy in normalized if abs(energy - ground) < threshold)
        success = ground_count / len(normalized) if normalized else 0.0
        return self.compute(success, t_anneal_us, p_target)

    def compare_solvers(
        self,
        results: Mapping[str, Mapping[str, Any]],
        ground_state_energy: float,
        tolerance: float = 1e-6,
    ) -> dict[str, dict[str, float]]:
        """Compute comparable TTS rows for named solver outputs."""
        comparison: dict[str, dict[str, float]] = {}
        for name, data in results.items():
            if not isinstance(name, str) or not name:
                raise ValueError("solver names must be non-empty strings")
            raw_energies = data.get("energies")
            if isinstance(raw_energies, (str, bytes)) or not isinstance(raw_energies, Sequence):
                raise ValueError(f"solver {name!r} must provide an energy sequence")
            comparison[name] = self.from_samples(
                raw_energies,
                ground_state_energy,
                t_anneal_us=_finite("t_anneal_us", data.get("t_anneal_us", 20.0)),
                tolerance=tolerance,
            )
        return comparison
