# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quantum and classical annealing solvers

"""Validated simulated-annealing and D-Wave solver adapters."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from sc_neurocore.bridges import annealing_backends as backends
from sc_neurocore.bridges.annealing_models import (
    BackendChoice,
    IsingModel,
    QUBOModel,
    validate_backend_choice,
)


_DEFAULT_CHAIN_STRENGTH = 2.0
_DEFAULT_NUM_READS = 1000
_DEFAULT_ANNEALING_TIME_US = 20.0


def _positive_int(name: str, value: int) -> int:
    """Return a positive integer or raise a field-specific error."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(name: str, value: float) -> float:
    """Return a finite positive float."""
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return numeric


def _finite_float(name: str, value: object) -> float:
    """Parse a finite numeric backend result."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"native solver returned invalid {name}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise RuntimeError(f"native solver returned non-finite {name}")
    return numeric


def _spin_sequence(name: str, value: object, size: int) -> list[int]:
    """Validate a native spin vector."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RuntimeError(f"native solver returned invalid {name}")
    spins = list(value)
    if len(spins) != size or any(spin not in {-1, 1} for spin in spins):
        raise RuntimeError(f"native solver returned invalid {name}")
    return [int(spin) for spin in spins]


class SimulatedAnnealer:
    """Metropolis simulated annealer with explicit backend selection."""

    def __init__(
        self,
        n_sweeps: int = 1000,
        beta_start: float = 0.1,
        beta_end: float = 10.0,
        seed: int = 42,
        *,
        backend: BackendChoice = "auto",
    ) -> None:
        """Configure a deterministic annealer."""
        self._n_sweeps = _positive_int("n_sweeps", n_sweeps)
        self._beta_start = _positive_float("beta_start", beta_start)
        self._beta_end = _positive_float("beta_end", beta_end)
        if self._beta_end < self._beta_start:
            raise ValueError("beta_end must be greater than or equal to beta_start")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self._seed = seed
        self._backend = validate_backend_choice(backend)
        self._rng = np.random.default_rng(seed)

    def solve_ising(self, model: IsingModel, num_reads: int = 10) -> dict[str, Any]:
        """Solve an Ising model and preserve a stable sample contract."""
        if not isinstance(model, IsingModel):
            raise ValueError("model must be an IsingModel")
        if model.n_qubits <= 0:
            raise ValueError("model must contain at least one qubit")
        reads = _positive_int("num_reads", num_reads)
        use_rust = self._backend == "rust" or (
            self._backend == "auto" and backends.HAS_RUST_QA and model.n_qubits > 10
        )
        if use_rust:
            return self._solve_ising_rust(model, reads)
        return self._solve_ising_python(model, reads)

    def _solve_ising_rust(self, model: IsingModel, num_reads: int) -> dict[str, Any]:
        """Execute and validate the native solver result."""
        h_indices = list(model.h)
        j_pairs = list(model.J)
        raw = backends.require_rust_annealer()(
            h_indices,
            [model.h[index] for index in h_indices],
            [pair[0] for pair in j_pairs],
            [pair[1] for pair in j_pairs],
            [model.J[pair] for pair in j_pairs],
            model.n_qubits,
            model.offset,
            self._n_sweeps,
            num_reads,
            self._beta_start,
            self._beta_end,
            self._seed,
        )
        best_vector = _spin_sequence("best_spins", raw.get("best_spins"), model.n_qubits)

        raw_samples = raw.get("samples", [])
        if isinstance(raw_samples, (str, bytes)) or not isinstance(raw_samples, Sequence):
            raise RuntimeError("native solver returned invalid samples")
        samples: list[dict[int, int]] = []
        for sample_index, sample in enumerate(raw_samples):
            vector = _spin_sequence(f"samples[{sample_index}]", sample, model.n_qubits)
            samples.append(dict(enumerate(vector)))

        raw_energies = raw.get("energies", [])
        if isinstance(raw_energies, (str, bytes)) or not isinstance(raw_energies, Sequence):
            raise RuntimeError("native solver returned invalid energies")
        energies = [
            _finite_float(f"energies[{index}]", energy) for index, energy in enumerate(raw_energies)
        ]
        if samples and len(samples) != len(energies):
            raise RuntimeError("native solver returned mismatched samples and energies")

        return {
            "best_spins": dict(enumerate(best_vector)),
            "best_energy": _finite_float("best_energy", raw.get("best_energy")),
            "energies": energies,
            "samples": samples,
            "n_sweeps": self._n_sweeps,
            "num_reads": num_reads,
            "backend": "rust",
        }

    def _solve_ising_python(self, model: IsingModel, num_reads: int) -> dict[str, Any]:
        """Execute the deterministic pure-Python Metropolis solver."""
        best_energy = float("inf")
        best_spins: dict[int, int] = {}
        all_energies: list[float] = []
        all_samples: list[dict[int, int]] = []

        for _ in range(num_reads):
            spins = {index: int(self._rng.choice((-1, 1))) for index in range(model.n_qubits)}
            energy = model.energy(spins, backend="python")
            for sweep in range(self._n_sweeps):
                exponent = sweep / max(self._n_sweeps - 1, 1)
                beta = self._beta_start * (self._beta_end / self._beta_start) ** exponent
                for qubit in range(model.n_qubits):
                    local_field = model.h.get(qubit, 0.0)
                    for (first, second), strength in model.J.items():
                        if first == qubit:
                            local_field += strength * spins.get(second, 1)
                        elif second == qubit:
                            local_field += strength * spins.get(first, 1)
                    delta_energy = -2.0 * spins[qubit] * local_field
                    if delta_energy < 0.0 or self._rng.random() < math.exp(-beta * delta_energy):
                        spins[qubit] *= -1
                        energy += delta_energy

            all_energies.append(energy)
            all_samples.append(dict(spins))
            if energy < best_energy:
                best_energy = energy
                best_spins = dict(spins)

        return {
            "best_spins": best_spins,
            "best_energy": best_energy,
            "energies": all_energies,
            "samples": all_samples,
            "n_sweeps": self._n_sweeps,
            "num_reads": num_reads,
            "backend": "python",
        }

    def solve_qubo(self, model: QUBOModel, num_reads: int = 10) -> dict[str, Any]:
        """Convert a QUBO to Ising, solve it, and map samples back to bits."""
        if not isinstance(model, QUBOModel):
            raise ValueError("model must be a QUBOModel")
        result = self.solve_ising(model.to_ising(), num_reads=num_reads)
        best_bits = {index: (spin + 1) // 2 for index, spin in result["best_spins"].items()}
        bit_samples = [
            {index: (spin + 1) // 2 for index, spin in sample.items()}
            for sample in result["samples"]
        ]
        return {
            "best_bits": best_bits,
            "best_energy": model.energy(best_bits),
            "energies": [model.energy(sample) for sample in bit_samples],
            "samples": bit_samples,
            "n_sweeps": self._n_sweeps,
            "num_reads": num_reads,
            "backend": result["backend"],
        }


class DWaveInterface:
    """Submit validated Ising models to D-Wave or use a local fallback."""

    def __init__(
        self,
        chain_strength: float = _DEFAULT_CHAIN_STRENGTH,
        num_reads: int = _DEFAULT_NUM_READS,
        annealing_time_us: float = _DEFAULT_ANNEALING_TIME_US,
    ) -> None:
        """Configure QPU sampling parameters."""
        self._chain_strength = _positive_float("chain_strength", chain_strength)
        self._num_reads = _positive_int("num_reads", num_reads)
        self._annealing_time_us = _positive_float("annealing_time_us", annealing_time_us)

    @property
    def available(self) -> bool:
        """Return whether both Ocean SDK components are importable."""
        return backends.HAS_DWAVE and backends.HAS_DIMOD

    def solve_ising(self, model: IsingModel) -> dict[str, Any]:
        """Submit to a QPU, or run a bounded local fallback when unavailable."""
        if not isinstance(model, IsingModel) or model.n_qubits <= 0:
            raise ValueError("model must be a non-empty IsingModel")
        if not self.available:
            result = SimulatedAnnealer().solve_ising(model, num_reads=min(self._num_reads, 20))
            result["backend"] = "simulated_annealing_fallback"
            return result

        dimod_module, sampler_type, composite_type = backends.require_dwave_components()
        bqm = dimod_module.BinaryQuadraticModel(model.h, model.J, model.offset, "SPIN")
        sampler = composite_type(sampler_type())
        response = sampler.sample(
            bqm,
            num_reads=self._num_reads,
            chain_strength=self._chain_strength,
            annealing_time=self._annealing_time_us,
        )
        best = getattr(response, "first", None)
        sample = getattr(best, "sample", None)
        energy = getattr(best, "energy", None)
        if not isinstance(sample, Mapping):
            raise RuntimeError("D-Wave response did not contain a best sample")
        best_spins = {int(index): int(spin) for index, spin in sample.items()}
        model.energy(best_spins, backend="python")
        info = getattr(response, "info", {})
        timing = info.get("timing", {}) if isinstance(info, Mapping) else {}
        return {
            "best_spins": best_spins,
            "best_energy": _finite_float("best_energy", energy),
            "num_reads": self._num_reads,
            "backend": "dwave_qpu",
            "timing": timing,
        }
