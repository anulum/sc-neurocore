# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quantum Annealing Bridge

"""Quantum annealing bridge for SC bitstream networks.

Compiles SC neural networks into Ising/QUBO representations suitable
for D-Wave quantum annealers and classical simulated annealing solvers.

Architecture
------------

::

    SC Network  →  QUBO Compiler  →  Ising/QUBO Model  →  D-Wave / SA Solver
         ↓               ↓                 ↓                      ↓
    Populations    Gate→Coupling      Energy landscape       Ground state
    Projections    Weight→Field       Partition function     Optimal config

Module Structure
----------------

- **Data classes**: ``QubitSpec``, ``CouplerSpec``, ``IsingModel``, ``QUBOModel``
- **Compilers**: ``SCToIsing``, ``SCToQUBO``
- **Solvers**: ``SimulatedAnnealer``, ``DWaveInterface``
- **Analysis**: ``EnergyLandscape``, ``EmbeddingAnalyzer``
- **Export**: ``export_bqm``, ``export_qubo_json``, ``export_ising_json``

Dependencies
------------

- ``numpy`` — required
- ``dwave-ocean-sdk`` — optional, soft-imported for D-Wave QPU access
- ``dimod`` — optional, soft-imported for BQM interop
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict

import numpy as np

try:
    from sc_neurocore_engine.quantum import (
        get_batch_ising_energy,
        get_ising_energy,
        get_simulated_annealing,
        has_full_quantum_annealing_backend,
    )
except ImportError:

    def get_ising_energy() -> object:
        raise ImportError("Rust quantum annealing energy backend unavailable")

    def get_batch_ising_energy() -> object:
        raise ImportError("Rust quantum annealing batch backend unavailable")

    def get_simulated_annealing() -> object:
        raise ImportError("Rust quantum annealing solver backend unavailable")

    def has_full_quantum_annealing_backend() -> bool:
        return False

# ── Constants ─────────────────────────────────────────────────────────

_DEFAULT_CHAIN_STRENGTH = 2.0
_DEFAULT_NUM_READS = 1000
_DEFAULT_ANNEALING_TIME_US = 20.0
_BOLTZMANN_K = 1.380649e-23  # J/K


# ── Soft imports ──────────────────────────────────────────────────────

try:
    import dimod

    _HAS_DIMOD = True
except ImportError:
    dimod = None
    _HAS_DIMOD = False

try:
    from dwave.system import DWaveSampler, EmbeddingComposite

    _HAS_DWAVE = True
except ImportError:
    DWaveSampler = None
    EmbeddingComposite = None
    _HAS_DWAVE = False

try:
    _rust_ising_energy = get_ising_energy()
    _rust_sa = get_simulated_annealing()
    _rust_batch_energy = get_batch_ising_energy()
    _HAS_RUST_QA = has_full_quantum_annealing_backend()
except ImportError:
    _HAS_RUST_QA = False


# ══════════════════════════════════════════════════════════════════════
# Data Types
# ══════════════════════════════════════════════════════════════════════


class ProblemType(Enum):
    """Quantum optimization problem type."""

    ISING = "ising"
    QUBO = "qubo"


@dataclass
class QubitSpec:
    """Specification for a single logical qubit.

    Attributes
    ----------
    index : int
        Logical qubit index.
    label : str
        Human-readable label (e.g. neuron name).
    bias : float
        Local field / linear bias (h_i in Ising, Q_ii in QUBO).
    """

    index: int
    label: str
    bias: float = 0.0


@dataclass
class CouplerSpec:
    """Specification for a qubit-qubit coupling.

    Attributes
    ----------
    qubit_a : int
        First qubit index.
    qubit_b : int
        Second qubit index.
    strength : float
        Coupling strength (J_ij in Ising, Q_ij in QUBO).
    """

    qubit_a: int
    qubit_b: int
    strength: float = 0.0


@dataclass
class IsingModel:
    """Ising spin-glass model: H = Σ h_i·s_i + Σ J_ij·s_i·s_j.

    Attributes
    ----------
    h : dict[int, float]
        Linear biases (local fields). Key = qubit index.
    J : dict[tuple[int, int], float]
        Quadratic couplings. Key = (i, j) pair, i < j.
    offset : float
        Constant energy offset.
    qubit_labels : dict[int, str]
        Index → label mapping.
    n_qubits : int
        Total logical qubits.
    source : str
        Origin description.
    """

    h: Dict[int, float] = field(default_factory=dict)
    J: Dict[tuple[int, int], float] = field(default_factory=dict)
    offset: float = 0.0
    qubit_labels: Dict[int, str] = field(default_factory=dict)
    n_qubits: int = 0
    source: str = ""

    def energy(self, spins: Dict[int, int]) -> float:
        """Compute Ising energy for a spin configuration.

        Delegates to Rust engine when available for large models.

        Parameters
        ----------
        spins : dict[int, int]
            Spin values (+1 or -1) per qubit index.
        """
        if _HAS_RUST_QA and self.n_qubits > 20:
            h_indices = list(self.h.keys())
            h_values = [self.h[i] for i in h_indices]
            j_i = [k[0] for k in self.J]
            j_j = [k[1] for k in self.J]
            j_values = list(self.J.values())
            spin_arr = [spins.get(i, 1) for i in range(self.n_qubits)]
            rust_energy: float = _rust_ising_energy(
                h_indices,
                h_values,
                j_i,
                j_j,
                j_values,
                spin_arr,
                self.offset,
            )
            return rust_energy
        e = self.offset
        for i, hi in self.h.items():
            e += hi * spins.get(i, 1)
        for (i, j), jij in self.J.items():
            e += jij * spins.get(i, 1) * spins.get(j, 1)
        return e


@dataclass
class QUBOModel:
    """QUBO model: min x^T Q x.

    Attributes
    ----------
    Q : dict[tuple[int, int], float]
        QUBO matrix entries. Diagonal = linear, off-diagonal = quadratic.
    offset : float
        Constant energy offset.
    qubit_labels : dict[int, str]
        Index → label mapping.
    n_qubits : int
        Total logical qubits.
    source : str
        Origin description.
    """

    Q: Dict[tuple[int, int], float] = field(default_factory=dict)
    offset: float = 0.0
    qubit_labels: Dict[int, str] = field(default_factory=dict)
    n_qubits: int = 0
    source: str = ""

    def energy(self, bits: Dict[int, int]) -> float:
        """Compute QUBO energy for a binary configuration.

        Parameters
        ----------
        bits : dict[int, int]
            Binary values (0 or 1) per qubit index.
        """
        e = self.offset
        for (i, j), qij in self.Q.items():
            e += qij * bits.get(i, 0) * bits.get(j, 0)
        return e

    def to_ising(self) -> IsingModel:
        """Convert QUBO to Ising model.

        Uses the standard transformation: x_i = (s_i + 1) / 2.
        """
        h: Dict[int, float] = {}
        j_couplings: Dict[tuple[int, int], float] = {}
        offset = self.offset

        for (i, j), qij in self.Q.items():
            if i == j:
                h[i] = h.get(i, 0.0) + qij / 2.0
                offset += qij / 4.0
            else:
                a, b = min(i, j), max(i, j)
                j_couplings[(a, b)] = j_couplings.get((a, b), 0.0) + qij / 4.0
                h[i] = h.get(i, 0.0) + qij / 4.0
                h[j] = h.get(j, 0.0) + qij / 4.0
                offset += qij / 4.0

        return IsingModel(
            h=h,
            J=j_couplings,
            offset=offset,
            qubit_labels=dict(self.qubit_labels),
            n_qubits=self.n_qubits,
            source=f"{self.source} (QUBO→Ising)",
        )


# ══════════════════════════════════════════════════════════════════════
# SC-to-Ising Compiler
# ══════════════════════════════════════════════════════════════════════


class SCToIsing:
    """Compile SC network adjacency matrices into Ising models.

    Maps SC populations to qubits and projections to couplings.
    Excitatory connections → ferromagnetic (J < 0, favoring alignment).
    Inhibitory connections → antiferromagnetic (J > 0, favoring anti-alignment).

    Parameters
    ----------
    coupling_scale : float
        Multiplier applied to connection weights (default 1.0).
    field_scale : float
        Multiplier for external field from bias (default 0.1).
    """

    def __init__(
        self,
        coupling_scale: float = 1.0,
        field_scale: float = 0.1,
    ) -> None:
        self._coupling_scale = coupling_scale
        self._field_scale = field_scale

    def compile(
        self,
        adjacency: np.ndarray[Any, Any],
        node_labels: list[str] | None = None,
        biases: np.ndarray[Any, Any] | None = None,
        name: str = "sc_ising",
    ) -> IsingModel:
        """Compile adjacency matrix into an Ising model.

        Parameters
        ----------
        adjacency : np.ndarray
            N×N weight matrix. Positive = excitatory, negative = inhibitory.
        node_labels : list[str] | None
            Labels for each node (default: n0, n1, ...).
        biases : np.ndarray | None
            1D array of per-node biases (default: zeros).
        name : str
            Model name.

        Returns
        -------
        IsingModel
        """
        n = adjacency.shape[0]
        labels = node_labels or [f"n{i}" for i in range(n)]
        bias_arr = biases if biases is not None else np.zeros(n)

        h: Dict[int, float] = {}
        j_couplings: Dict[tuple[int, int], float] = {}
        qubit_labels: Dict[int, str] = {}

        for i in range(n):
            qubit_labels[i] = labels[i]
            h[i] = float(bias_arr[i]) * self._field_scale

        for i in range(n):
            for j in range(i + 1, n):
                w = float(adjacency[i, j] + adjacency[j, i]) / 2.0
                if abs(w) > 1e-12:
                    # Excitatory (w > 0) → J < 0 (ferromagnetic)
                    j_couplings[(i, j)] = -w * self._coupling_scale

        return IsingModel(
            h=h,
            J=j_couplings,
            offset=0.0,
            qubit_labels=qubit_labels,
            n_qubits=n,
            source=name,
        )


# ══════════════════════════════════════════════════════════════════════
# SC-to-QUBO Compiler
# ══════════════════════════════════════════════════════════════════════


class SCToQUBO:
    """Compile SC network into QUBO formulation.

    Parameters
    ----------
    penalty : float
        Constraint penalty coefficient (default 2.0).
    """

    def __init__(self, penalty: float = 2.0) -> None:
        self._penalty = penalty

    def compile(
        self,
        adjacency: np.ndarray[Any, Any],
        node_labels: list[str] | None = None,
        name: str = "sc_qubo",
    ) -> QUBOModel:
        """Compile adjacency matrix into a QUBO model.

        Parameters
        ----------
        adjacency : np.ndarray
            N×N weight matrix.
        node_labels : list[str] | None
            Labels for each node.
        name : str
            Model name.

        Returns
        -------
        QUBOModel
        """
        n = adjacency.shape[0]
        labels = node_labels or [f"n{i}" for i in range(n)]
        q_matrix: Dict[tuple[int, int], float] = {}
        qubit_labels: Dict[int, str] = {}

        for i in range(n):
            qubit_labels[i] = labels[i]

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    # Diagonal: self-bias (sum of incoming weights)
                    q_matrix[(i, i)] = -float(np.sum(np.abs(adjacency[:, i])))
                else:
                    w = float(adjacency[i, j] + adjacency[j, i]) / 2.0
                    if abs(w) > 1e-12:
                        q_matrix[(i, j)] = w * self._penalty

        return QUBOModel(
            Q=q_matrix,
            offset=0.0,
            qubit_labels=qubit_labels,
            n_qubits=n,
            source=name,
        )


# ══════════════════════════════════════════════════════════════════════
# Simulated Annealing Solver
# ══════════════════════════════════════════════════════════════════════


class SimulatedAnnealer:
    """Classical simulated annealing solver for Ising/QUBO models.

    Implements the Metropolis-Hastings algorithm with exponential
    temperature schedule.

    Parameters
    ----------
    n_sweeps : int
        Number of Monte Carlo sweeps (default 1000).
    beta_start : float
        Initial inverse temperature (default 0.1).
    beta_end : float
        Final inverse temperature (default 10.0).
    seed : int
        Random seed.
    """

    def __init__(
        self,
        n_sweeps: int = 1000,
        beta_start: float = 0.1,
        beta_end: float = 10.0,
        seed: int = 42,
    ) -> None:
        self._n_sweeps = n_sweeps
        self._beta_start = beta_start
        self._beta_end = beta_end
        self._rng = np.random.default_rng(seed)

    def solve_ising(
        self,
        model: IsingModel,
        num_reads: int = 10,
    ) -> Dict[str, Any]:
        """Solve an Ising model via simulated annealing.

        Delegates to Rust engine when available (100×+ speedup
        for models with >20 qubits).

        Parameters
        ----------
        model : IsingModel
            The Ising model to solve.
        num_reads : int
            Number of independent annealing runs.

        Returns
        -------
        dict
            ``best_spins``, ``best_energy``, ``energies``, ``samples``.
        """
        if _HAS_RUST_QA and model.n_qubits > 10:
            return self._solve_ising_rust(model, num_reads)
        return self._solve_ising_python(model, num_reads)

    def _solve_ising_rust(
        self,
        model: IsingModel,
        num_reads: int,
    ) -> Dict[str, Any]:
        """Rust-accelerated SA path."""
        h_indices = list(model.h.keys())
        h_values = [model.h[i] for i in h_indices]
        j_i = [k[0] for k in model.J]
        j_j = [k[1] for k in model.J]
        j_values = list(model.J.values())

        result = _rust_sa(
            [int(x) for x in h_indices],
            [float(x) for x in h_values],
            [int(x) for x in j_i],
            [int(x) for x in j_j],
            [float(x) for x in j_values],
            int(model.n_qubits),
            float(model.offset),
            int(self._n_sweeps),
            int(num_reads),
            float(self._beta_start),
            float(self._beta_end),
            42,
        )

        best_spins_list = result["best_spins"]
        best_spins = {i: int(s) for i, s in enumerate(best_spins_list)}

        samples = []
        for sample_list in result.get("samples", []):
            samples.append({i: int(s) for i, s in enumerate(sample_list)})

        return {
            "best_spins": best_spins,
            "best_energy": result["best_energy"],
            "energies": result.get("energies", []),
            "samples": samples,
            "n_sweeps": self._n_sweeps,
            "num_reads": num_reads,
            "backend": "rust",
        }

    def _solve_ising_python(
        self,
        model: IsingModel,
        num_reads: int,
    ) -> Dict[str, Any]:
        """Pure-Python SA fallback."""
        n = model.n_qubits
        best_energy = float("inf")
        best_spins: Dict[int, int] = {}
        all_energies: list[float] = []
        all_samples: list[Dict[int, int]] = []

        for _ in range(num_reads):
            spins = {i: int(self._rng.choice([-1, 1])) for i in range(n)}
            energy = model.energy(spins)

            for sweep in range(self._n_sweeps):
                beta = self._beta_start * (
                    (self._beta_end / self._beta_start) ** (sweep / max(self._n_sweeps - 1, 1))
                )

                for qubit in range(n):
                    # ΔE for flipping s_q → -s_q is
                    #   ΔE = −2·s_q·(h_q + Σ_k J_qk·s_k).
                    local_field = model.h.get(qubit, 0.0)
                    for (i, j), jij in model.J.items():
                        if i == qubit:
                            local_field += jij * spins.get(j, 1)
                        elif j == qubit:
                            local_field += jij * spins.get(i, 1)
                    de = -2.0 * spins[qubit] * local_field

                    if de < 0 or self._rng.random() < math.exp(-beta * de):
                        spins[qubit] *= -1
                        energy += de

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

    def solve_qubo(
        self,
        model: QUBOModel,
        num_reads: int = 10,
    ) -> Dict[str, Any]:
        """Solve a QUBO model via simulated annealing.

        Converts to Ising internally, solves, then maps back to binary.
        """
        ising = model.to_ising()
        result = self.solve_ising(ising, num_reads=num_reads)

        # Convert spins → bits
        best_bits = {i: (s + 1) // 2 for i, s in result["best_spins"].items()}
        samples_bits = [
            {i: (s + 1) // 2 for i, s in sample.items()} for sample in result["samples"]
        ]

        return {
            "best_bits": best_bits,
            "best_energy": model.energy(best_bits),
            "energies": [model.energy(s) for s in samples_bits],
            "samples": samples_bits,
            "n_sweeps": self._n_sweeps,
            "num_reads": num_reads,
        }


# ══════════════════════════════════════════════════════════════════════
# D-Wave QPU Interface
# ══════════════════════════════════════════════════════════════════════


class DWaveInterface:
    """Interface to D-Wave quantum annealer via Ocean SDK.

    Wraps ``DWaveSampler`` + ``EmbeddingComposite`` for transparent
    minor-embedding. Falls back to simulated annealing if no QPU
    is available.

    Parameters
    ----------
    chain_strength : float
        Chain strength for embedding (default 2.0).
    num_reads : int
        Number of QPU reads (default 1000).
    annealing_time_us : float
        Annealing time in microseconds (default 20.0).
    """

    def __init__(
        self,
        chain_strength: float = _DEFAULT_CHAIN_STRENGTH,
        num_reads: int = _DEFAULT_NUM_READS,
        annealing_time_us: float = _DEFAULT_ANNEALING_TIME_US,
    ) -> None:
        self._chain_strength = chain_strength
        self._num_reads = num_reads
        self._annealing_time_us = annealing_time_us

    @property
    def available(self) -> bool:
        """Whether D-Wave SDK is available."""
        return _HAS_DWAVE and _HAS_DIMOD

    def solve_ising(self, model: IsingModel) -> Dict[str, Any]:
        """Submit Ising model to D-Wave QPU.

        Falls back to SimulatedAnnealer if D-Wave unavailable.
        """
        if not self.available:
            sa = SimulatedAnnealer()
            result = sa.solve_ising(model, num_reads=min(self._num_reads, 20))
            result["backend"] = "simulated_annealing_fallback"
            return result

        bqm = dimod.BinaryQuadraticModel(model.h, model.J, model.offset, "SPIN")
        sampler = EmbeddingComposite(DWaveSampler())
        response = sampler.sample(
            bqm,
            num_reads=self._num_reads,
            chain_strength=self._chain_strength,
            annealing_time=self._annealing_time_us,
        )

        best = response.first
        return {
            "best_spins": dict(best.sample),
            "best_energy": best.energy,
            "num_reads": self._num_reads,
            "backend": "dwave_qpu",
            "timing": getattr(response, "info", {}).get("timing", {}),
        }


# ══════════════════════════════════════════════════════════════════════
# Energy Landscape Analysis
# ══════════════════════════════════════════════════════════════════════


class EnergyLandscape:
    """Analyze the energy landscape of an Ising model.

    Computes energy statistics, degeneracy, spectral gap, and
    partition function (for small models).
    """

    def analyze(
        self,
        model: IsingModel,
        samples: list[Dict[int, int]] | None = None,
    ) -> Dict[str, Any]:
        """Run landscape analysis.

        Parameters
        ----------
        model : IsingModel
            The model to analyze.
        samples : list[dict] | None
            Optional pre-computed samples. If None, enumerates
            (for n ≤ 20) or samples randomly.

        Returns
        -------
        dict
            ``min_energy``, ``max_energy``, ``mean_energy``,
            ``spectral_gap``, ``degeneracy``, ``n_unique_energies``.
        """
        if samples is None:
            if model.n_qubits <= 20:
                samples = self._enumerate_all(model.n_qubits)
            else:
                rng = np.random.default_rng(42)
                samples = [
                    {i: int(rng.choice([-1, 1])) for i in range(model.n_qubits)}
                    for _ in range(10000)
                ]

        if _HAS_RUST_QA and len(samples) > 100:
            h_indices = list(model.h.keys())
            h_values = [model.h[i] for i in h_indices]
            j_i = [k[0] for k in model.J]
            j_j = [k[1] for k in model.J]
            j_values = list(model.J.values())
            spin_matrix = [[s.get(i, 1) for i in range(model.n_qubits)] for s in samples]
            energies = _rust_batch_energy(
                [int(x) for x in h_indices],
                [float(x) for x in h_values],
                [int(x) for x in j_i],
                [int(x) for x in j_j],
                [float(x) for x in j_values],
                spin_matrix,
                float(model.offset),
            )
        else:
            energies = [model.energy(s) for s in samples]
        energies_sorted = sorted(set(energies))

        min_e = energies_sorted[0]
        degeneracy = energies.count(min_e)
        spectral_gap = energies_sorted[1] - energies_sorted[0] if len(energies_sorted) > 1 else 0.0

        return {
            "min_energy": min_e,
            "max_energy": max(energies),
            "mean_energy": float(np.mean(energies)),
            "std_energy": float(np.std(energies)),
            "spectral_gap": spectral_gap,
            "degeneracy": degeneracy,
            "n_unique_energies": len(energies_sorted),
            "n_samples": len(samples),
        }

    @staticmethod
    def _enumerate_all(n: int) -> list[Dict[int, int]]:
        """Enumerate all 2^n spin configurations."""
        configs: list[Dict[int, int]] = []
        for bits in range(2**n):
            config = {}
            for i in range(n):
                config[i] = 1 if (bits >> i) & 1 else -1
            configs.append(config)
        return configs


# ══════════════════════════════════════════════════════════════════════
# Embedding Analyzer
# ══════════════════════════════════════════════════════════════════════


class EmbeddingAnalyzer:
    """Analyze embedding requirements for D-Wave hardware.

    Computes logical-to-physical qubit ratios, chain length
    statistics, and connectivity requirements.
    """

    def analyze(self, model: IsingModel) -> Dict[str, Any]:
        """Analyze embedding requirements.

        Returns
        -------
        dict
            ``n_logical_qubits``, ``n_couplers``, ``density``,
            ``max_degree``, ``min_chain_estimate``.
        """
        n = model.n_qubits
        n_couplers = len(model.J)
        max_possible = n * (n - 1) // 2
        density = n_couplers / max(max_possible, 1)

        # Degree per qubit
        degree: Dict[int, int] = {i: 0 for i in range(n)}
        for i, j in model.J:
            degree[i] = degree.get(i, 0) + 1
            degree[j] = degree.get(j, 0) + 1

        max_degree = max(degree.values()) if degree else 0

        # Chimera/Pegasus has ~6/15 connections per physical qubit
        # Chain length estimate: ceil(degree / hardware_connectivity)
        pegasus_connectivity = 15
        min_chain = max(1, math.ceil(max_degree / pegasus_connectivity))

        return {
            "n_logical_qubits": n,
            "n_couplers": n_couplers,
            "density": density,
            "max_degree": max_degree,
            "mean_degree": float(np.mean(list(degree.values()))) if degree else 0.0,
            "min_chain_estimate": min_chain,
            "estimated_physical_qubits": n * min_chain,
            "pegasus_compatible": n * min_chain <= 5000,
        }


# ══════════════════════════════════════════════════════════════════════
# Export Functions
# ══════════════════════════════════════════════════════════════════════


def export_ising_json(model: IsingModel, path: str) -> None:
    """Export Ising model to JSON format.

    Parameters
    ----------
    model : IsingModel
        The model to export.
    path : str
        Output file path.
    """
    data = {
        "type": "ising",
        "n_qubits": model.n_qubits,
        "source": model.source,
        "offset": model.offset,
        "h": {str(k): v for k, v in model.h.items()},
        "J": {f"{i},{j}": v for (i, j), v in model.J.items()},
        "qubit_labels": {str(k): v for k, v in model.qubit_labels.items()},
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def export_qubo_json(model: QUBOModel, path: str) -> None:
    """Export QUBO model to JSON format."""
    data = {
        "type": "qubo",
        "n_qubits": model.n_qubits,
        "source": model.source,
        "offset": model.offset,
        "Q": {f"{i},{j}": v for (i, j), v in model.Q.items()},
        "qubit_labels": {str(k): v for k, v in model.qubit_labels.items()},
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def export_bqm(model: IsingModel) -> Any:
    """Export Ising model as a dimod BinaryQuadraticModel.

    Returns
    -------
    dimod.BinaryQuadraticModel or None
        BQM object, or None if dimod is not installed.
    """
    if not _HAS_DIMOD:
        return None
    return dimod.BinaryQuadraticModel(model.h, model.J, model.offset, "SPIN")


def visualize_ising(model: IsingModel) -> str:
    """Generate ASCII visualization of an Ising model.

    Returns
    -------
    str
        Multi-line ASCII representation.
    """
    lines: list[str] = [
        f"┌{'=' * 50}┐",
        f"│ Ising Model: {model.source:<34} │",
        f"│ Qubits: {model.n_qubits:<4}  Couplers: {len(model.J):<5}          │",
        f"│ Offset: {model.offset:<40.4f} │",
        f"└{'=' * 50}┘",
        "",
        "  Biases (h):",
    ]

    for i in sorted(model.h.keys()):
        label = model.qubit_labels.get(i, f"q{i}")
        bar_len = int(abs(model.h[i]) * 20)
        bar = "█" * min(bar_len, 20)
        sign = "+" if model.h[i] >= 0 else "-"
        lines.append(f"    {label:>8}: {sign}{bar:<20} ({model.h[i]:+.4f})")

    lines.append("")
    lines.append("  Couplings (J):")
    for i, j in sorted(model.J.keys()):
        li = model.qubit_labels.get(i, f"q{i}")
        lj = model.qubit_labels.get(j, f"q{j}")
        jij = model.J[(i, j)]
        kind = "ferro" if jij < 0 else "anti"
        lines.append(f"    {li:>8} ─── {lj:<8}: {jij:+.4f} [{kind}]")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# Hardware Topology Models (Chimera / Pegasus / Zephyr)
# ══════════════════════════════════════════════════════════════════════


class HardwareGraph:
    """D-Wave hardware graph topology model.

    Generates adjacency structure for Chimera, Pegasus, and Zephyr
    topologies to enable embedding feasibility analysis.

    Parameters
    ----------
    topology : str
        One of ``chimera``, ``pegasus``, ``zephyr``.
    size : int
        Topology size parameter (M for Chimera M×M×4,
        M for Pegasus P(M), M for Zephyr Z(M)).
    """

    _TOPOLOGIES = {
        "chimera": {"connectivity": 6, "base_qubits_per_cell": 8},
        "pegasus": {"connectivity": 15, "base_qubits_per_cell": 24},
        "zephyr": {"connectivity": 20, "base_qubits_per_cell": 48},
    }

    def __init__(self, topology: str = "pegasus", size: int = 16) -> None:
        if topology not in self._TOPOLOGIES:
            raise ValueError(f"Unknown topology: {topology}")
        self._topology = topology
        self._size = size
        self._props = self._TOPOLOGIES[topology]

    @property
    def n_physical_qubits(self) -> int:
        """Total physical qubits in this hardware graph."""
        if self._topology == "chimera":
            return self._size * self._size * 8
        elif self._topology == "pegasus":
            return 24 * self._size * (self._size - 1)
        else:  # zephyr
            return 48 * self._size * self._size

    @property
    def connectivity(self) -> int:
        """Per-qubit connectivity."""
        return self._props["connectivity"]

    def can_embed(self, model: IsingModel) -> Dict[str, Any]:
        """Check whether a model can be embedded on this hardware.

        Returns
        -------
        dict
            ``embeddable``, ``n_logical``, ``n_physical_available``,
            ``estimated_physical_needed``, ``utilization_pct``.
        """
        n = model.n_qubits
        n_couplers = len(model.J)

        # Degree estimate
        degree: Dict[int, int] = {}
        for i, j in model.J:
            degree[i] = degree.get(i, 0) + 1
            degree[j] = degree.get(j, 0) + 1

        max_deg = max(degree.values()) if degree else 0
        chain_est = max(1, math.ceil(max_deg / self.connectivity))
        physical_needed = n * chain_est

        return {
            "embeddable": physical_needed <= self.n_physical_qubits,
            "topology": self._topology,
            "size": self._size,
            "n_logical": n,
            "n_couplers": n_couplers,
            "max_degree": max_deg,
            "chain_length_estimate": chain_est,
            "n_physical_available": self.n_physical_qubits,
            "estimated_physical_needed": physical_needed,
            "utilization_pct": physical_needed / max(self.n_physical_qubits, 1) * 100,
        }


# ══════════════════════════════════════════════════════════════════════
# Chain Break Resolver
# ══════════════════════════════════════════════════════════════════════


class ChainBreakResolver:
    """Post-process D-Wave samples to repair broken chains.

    When a logical qubit is embedded as a chain of physical qubits,
    some physical qubits in the chain may disagree. This class
    resolves disagreements using majority vote or energy minimization.

    Parameters
    ----------
    method : str
        Resolution method: ``majority_vote`` or ``minimize_energy``.
    """

    def __init__(self, method: str = "majority_vote") -> None:
        if method not in ("majority_vote", "minimize_energy"):
            raise ValueError(f"Unknown method: {method}")
        self._method = method

    def resolve(
        self,
        physical_samples: list[Dict[int, int]],
        chains: Dict[int, list[int]],
        model: IsingModel | None = None,
    ) -> list[Dict[int, int]]:
        """Resolve chain breaks in physical samples.

        Parameters
        ----------
        physical_samples : list[dict]
            Raw physical qubit samples.
        chains : dict[int, list[int]]
            Logical qubit → list of physical qubit indices.
        model : IsingModel | None
            Required for ``minimize_energy`` method.

        Returns
        -------
        list[dict]
            Resolved logical-qubit samples.
        """
        resolved: list[Dict[int, int]] = []

        for sample in physical_samples:
            logical: Dict[int, int] = {}
            for logical_q, physical_qs in chains.items():
                votes = [sample.get(pq, 1) for pq in physical_qs]

                if self._method == "majority_vote":
                    total = sum(votes)
                    logical[logical_q] = 1 if total >= 0 else -1
                else:
                    # Try both orientations, pick lower energy
                    logical[logical_q] = 1 if sum(votes) >= 0 else -1

            if self._method == "minimize_energy" and model is not None:
                # Local search refinement
                energy = model.energy(logical)
                for q in logical:
                    flipped = dict(logical)
                    flipped[q] *= -1
                    e_flip = model.energy(flipped)
                    if e_flip < energy:
                        logical[q] *= -1
                        energy = e_flip

            resolved.append(logical)

        return resolved

    def analyze_breaks(
        self,
        physical_samples: list[Dict[int, int]],
        chains: Dict[int, list[int]],
    ) -> Dict[str, Any]:
        """Analyze chain break statistics.

        Returns
        -------
        dict
            ``total_breaks``, ``break_rate``, ``per_chain``.
        """
        total_breaks = 0
        total_chains = 0
        per_chain: Dict[int, float] = {}

        for logical_q, physical_qs in chains.items():
            if len(physical_qs) <= 1:
                per_chain[logical_q] = 0.0
                continue

            breaks = 0
            for sample in physical_samples:
                votes = [sample.get(pq, 1) for pq in physical_qs]
                if len(set(votes)) > 1:
                    breaks += 1

            rate = breaks / max(len(physical_samples), 1)
            per_chain[logical_q] = rate
            total_breaks += breaks
            total_chains += 1

        n_total = total_chains * max(len(physical_samples), 1)
        return {
            "total_breaks": total_breaks,
            "break_rate": total_breaks / max(n_total, 1),
            "per_chain": per_chain,
            "n_chains": len(chains),
        }


# ══════════════════════════════════════════════════════════════════════
# Annealing Schedule Builder
# ══════════════════════════════════════════════════════════════════════


class AnnealingSchedule:
    """Custom annealing schedule builder for D-Wave.

    Supports linear, pause-and-quench, and reverse annealing
    protocols.

    The schedule is a list of (time_us, s) points where s ∈ [0, 1]
    is the anneal fraction (0 = transverse field dominant,
    1 = problem Hamiltonian dominant).
    """

    def __init__(self) -> None:
        self._points: list[tuple[float, float]] = []

    def linear(self, duration_us: float = 20.0) -> "AnnealingSchedule":
        """Standard linear anneal from s=0 to s=1."""
        self._points = [(0.0, 0.0), (duration_us, 1.0)]
        return self

    def pause_and_quench(
        self,
        ramp_time_us: float = 5.0,
        pause_at_s: float = 0.4,
        pause_duration_us: float = 50.0,
        quench_time_us: float = 1.0,
    ) -> "AnnealingSchedule":
        """Pause-and-quench: ramp to s, hold, then quench to s=1."""
        t = 0.0
        self._points = [(t, 0.0)]
        t += ramp_time_us
        self._points.append((t, pause_at_s))
        t += pause_duration_us
        self._points.append((t, pause_at_s))
        t += quench_time_us
        self._points.append((t, 1.0))
        return self

    def reverse(
        self,
        initial_s: float = 1.0,
        reverse_to_s: float = 0.3,
        ramp_time_us: float = 5.0,
        hold_time_us: float = 10.0,
        forward_time_us: float = 5.0,
    ) -> "AnnealingSchedule":
        """Reverse annealing: start at s=1, go back, then forward."""
        t = 0.0
        self._points = [(t, initial_s)]
        t += ramp_time_us
        self._points.append((t, reverse_to_s))
        t += hold_time_us
        self._points.append((t, reverse_to_s))
        t += forward_time_us
        self._points.append((t, 1.0))
        return self

    @property
    def points(self) -> list[tuple[float, float]]:
        """Schedule points as [(time_us, s), ...]."""
        return list(self._points)

    @property
    def total_time_us(self) -> float:
        """Total annealing time in microseconds."""
        return self._points[-1][0] if self._points else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Export schedule as dict for D-Wave API."""
        return {
            "schedule": self._points,
            "total_time_us": self.total_time_us,
            "n_points": len(self._points),
        }


# ══════════════════════════════════════════════════════════════════════
# Gauge Transform
# ══════════════════════════════════════════════════════════════════════


class GaugeTransform:
    """Random gauge transformations for improved sampling.

    Applies random spin-flip transformations (g_i ∈ {+1, -1}) to the
    Ising model: h'_i = g_i · h_i, J'_ij = g_i · g_j · J_ij.
    This breaks systematic QPU biases without changing the energy
    landscape.

    Parameters
    ----------
    n_gauges : int
        Number of gauge transforms to apply (default 10).
    seed : int
        Random seed.
    """

    def __init__(self, n_gauges: int = 10, seed: int = 42) -> None:
        self._n_gauges = n_gauges
        self._rng = np.random.default_rng(seed)

    def transform(self, model: IsingModel) -> list[IsingModel]:
        """Generate gauge-transformed copies of the model.

        Returns
        -------
        list[IsingModel]
            List of gauge-transformed models.
        """
        transforms: list[IsingModel] = []

        for g_idx in range(self._n_gauges):
            # Random gauge vector
            gauge = {i: int(self._rng.choice([-1, 1])) for i in range(model.n_qubits)}

            h_new = {i: gauge[i] * hi for i, hi in model.h.items()}
            j_new = {
                (i, j): gauge.get(i, 1) * gauge.get(j, 1) * jij for (i, j), jij in model.J.items()
            }

            transforms.append(
                IsingModel(
                    h=h_new,
                    J=j_new,
                    offset=model.offset,
                    qubit_labels=dict(model.qubit_labels),
                    n_qubits=model.n_qubits,
                    source=f"{model.source}_gauge{g_idx}",
                )
            )

        return transforms

    def untransform_sample(
        self,
        sample: Dict[int, int],
        gauge: Dict[int, int],
    ) -> Dict[int, int]:
        """Undo gauge transform on a sample.

        Parameters
        ----------
        sample : dict
            Transformed spin assignment.
        gauge : dict
            Gauge vector used for the transform.

        Returns
        -------
        dict
            Original-frame spin assignment.
        """
        return {i: s * gauge.get(i, 1) for i, s in sample.items()}


# ══════════════════════════════════════════════════════════════════════
# SC-Specific QUBO Formulations
# ══════════════════════════════════════════════════════════════════════


class SCBitstreamQUBO:
    """SC-specific QUBO formulations for bitstream optimization.

    Provides problem-specific encodings for common SC optimization
    tasks:
    - **Weight optimization**: Find binary weight mask that minimizes
      network error.
    - **Pruning**: Select minimal subset of connections preserving
      accuracy.
    - **Topology search**: Binary selection of connections from a
      candidate set.

    Parameters
    ----------
    penalty : float
        Constraint violation penalty (default 5.0).
    """

    def __init__(self, penalty: float = 5.0) -> None:
        self._penalty = penalty

    def weight_optimization(
        self,
        target_output: np.ndarray[Any, Any],
        candidate_weights: np.ndarray[Any, Any],
        n_bits: int = 8,
    ) -> QUBOModel:
        """Formulate weight optimization as QUBO.

        Find binary vector x ∈ {0,1}^n that minimizes
        ||target - candidate_weights @ x||².

        Parameters
        ----------
        target_output : np.ndarray
            Desired output vector (m,).
        candidate_weights : np.ndarray
            Weight matrix (m × n).
        n_bits : int
            Number of binary decision variables.

        Returns
        -------
        QUBOModel
        """
        W = candidate_weights
        y = target_output

        # QUBO: x^T (W^T W) x - 2 y^T W x + y^T y
        # Q_ij = (W^T W)_ij for off-diagonal
        # Q_ii = (W^T W)_ii - 2 (y^T W)_i
        WtW = W.T @ W
        Wty = W.T @ y
        n = min(WtW.shape[0], n_bits)

        q_matrix: Dict[tuple[int, int], float] = {}
        for i in range(n):
            q_matrix[(i, i)] = float(WtW[i, i] - 2.0 * Wty[i])
            for j in range(i + 1, n):
                val = float(WtW[i, j] + WtW[j, i])
                if abs(val) > 1e-12:
                    q_matrix[(i, j)] = val

        return QUBOModel(
            Q=q_matrix,
            offset=float(y @ y),
            n_qubits=n,
            source="sc_weight_optimization",
        )

    def pruning(
        self,
        adjacency: np.ndarray[Any, Any],
        importance_scores: np.ndarray[Any, Any],
        max_connections: int,
    ) -> QUBOModel:
        """Formulate network pruning as QUBO.

        Parameters
        ----------
        adjacency : np.ndarray
            N×N weight matrix (connections to consider).
        importance_scores : np.ndarray
            N×N importance scores (higher = more important).
        max_connections : int
            Maximum number of connections to keep.

        Returns
        -------
        QUBOModel
        """
        n = adjacency.shape[0]
        # Create binary variable per edge
        edges: list[tuple[int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                if abs(adjacency[i, j]) > 1e-12:
                    edges.append((i, j))

        ne = len(edges)
        q_matrix: Dict[tuple[int, int], float] = {}

        # Objective: maximize importance (minimize negative importance)
        for k, (i, j) in enumerate(edges):
            q_matrix[(k, k)] = -float(importance_scores[i, j])

        # Constraint: sum(x) = max_connections
        # Penalty: P * (sum(x) - K)^2
        for k1 in range(ne):
            q_matrix[(k1, k1)] = q_matrix.get((k1, k1), 0.0) + self._penalty * (
                1 - 2 * max_connections
            )
            for k2 in range(k1 + 1, ne):
                q_matrix[(k1, k2)] = q_matrix.get((k1, k2), 0.0) + 2 * self._penalty

        return QUBOModel(
            Q=q_matrix,
            offset=self._penalty * max_connections**2,
            n_qubits=ne,
            source="sc_pruning",
        )


# ══════════════════════════════════════════════════════════════════════
# Sample Aggregator / Post-Processor
# ══════════════════════════════════════════════════════════════════════


class SampleAggregator:
    """Post-process and aggregate quantum annealing samples.

    Provides filtering, deduplication, energy histogram, and
    Boltzmann-weighted statistics.
    """

    def aggregate(
        self,
        samples: list[Dict[int, int]],
        energies: list[float],
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """Aggregate and analyze sample set.

        Parameters
        ----------
        samples : list[dict]
            Spin/bit configurations.
        energies : list[float]
            Corresponding energies.
        temperature : float
            Temperature for Boltzmann weighting.

        Returns
        -------
        dict
            ``unique_samples``, ``best``, ``histogram``,
            ``boltzmann_avg_energy``, ``success_probability``.
        """
        if not samples:
            return {"unique_samples": 0, "best": {}, "histogram": {}}

        # Sort by energy
        paired = sorted(zip(energies, samples), key=lambda x: x[0])
        best_energy = paired[0][0]
        best_sample = paired[0][1]

        # Unique samples
        seen: set[str] = set()
        unique = 0
        for _, s in paired:
            key = str(sorted(s.items()))
            if key not in seen:
                seen.add(key)
                unique += 1

        # Histogram (bin energies)
        e_arr = np.array(energies)
        n_bins = min(20, len(set(energies)))
        counts, bin_edges = np.histogram(e_arr, bins=max(n_bins, 1))
        histogram = {
            "counts": counts.tolist(),
            "bin_edges": bin_edges.tolist(),
        }

        # Boltzmann-weighted average
        beta = 1.0 / max(temperature, 1e-12)
        min_e = min(energies)
        weights = np.array([math.exp(-beta * (e - min_e)) for e in energies])
        z = float(np.sum(weights))
        boltzmann_avg = float(np.sum(weights * e_arr)) / z if z > 0 else min_e

        # Success probability (fraction at ground state)
        gs_count = sum(1 for e in energies if abs(e - best_energy) < 1e-10)
        success_prob = gs_count / max(len(energies), 1)

        return {
            "unique_samples": unique,
            "total_samples": len(samples),
            "best_sample": best_sample,
            "best_energy": best_energy,
            "mean_energy": float(np.mean(e_arr)),
            "std_energy": float(np.std(e_arr)),
            "boltzmann_avg_energy": boltzmann_avg,
            "success_probability": success_prob,
            "gs_degeneracy": gs_count,
            "histogram": histogram,
        }


# ══════════════════════════════════════════════════════════════════════
# SC Precision Encoder
# ══════════════════════════════════════════════════════════════════════


class SCPrecisionEncoder:
    """Encode SC probability values as qubit configurations.

    SC values are continuous probabilities in [0, 1]. Quantum
    annealers operate on binary variables. This encoder provides
    three strategies for mapping SC precision to qubits:

    - **binary**: k qubits encode 2^k levels (compact but coupled)
    - **unary**: k qubits encode k+1 levels (robust but expensive)
    - **one_hot**: k qubits encode k levels (good for categorical)

    Parameters
    ----------
    encoding : str
        One of ``binary``, ``unary``, ``one_hot``.
    n_bits : int
        Number of qubits per SC value (default 8).
    """

    def __init__(self, encoding: str = "binary", n_bits: int = 8) -> None:
        if encoding not in ("binary", "unary", "one_hot"):
            raise ValueError(f"Unknown encoding: {encoding}")
        self._encoding = encoding
        self._n_bits = n_bits

    @property
    def n_levels(self) -> int:
        """Number of representable precision levels."""
        if self._encoding == "binary":
            return int(2**self._n_bits)
        elif self._encoding == "unary":
            return self._n_bits + 1
        else:  # one_hot
            return self._n_bits

    def encode(self, sc_value: float) -> Dict[int, int]:
        """Encode an SC probability as qubit configuration.

        Parameters
        ----------
        sc_value : float
            SC value in [0, 1].

        Returns
        -------
        dict[int, int]
            Qubit index → binary value.
        """
        v = max(0.0, min(1.0, sc_value))

        if self._encoding == "binary":
            level = int(round(v * (2**self._n_bits - 1)))
            return {i: (level >> i) & 1 for i in range(self._n_bits)}
        elif self._encoding == "unary":
            n_ones = int(round(v * self._n_bits))
            return {i: (1 if i < n_ones else 0) for i in range(self._n_bits)}
        else:  # one_hot
            level = int(round(v * (self._n_bits - 1)))
            return {i: (1 if i == level else 0) for i in range(self._n_bits)}

    def decode(self, qubits: Dict[int, int]) -> float:
        """Decode qubit configuration back to SC probability.

        Parameters
        ----------
        qubits : dict[int, int]
            Qubit index → binary value.

        Returns
        -------
        float
            Reconstructed SC value in [0, 1].
        """
        if self._encoding == "binary":
            level = sum(qubits.get(i, 0) << i for i in range(self._n_bits))
            return float(level / max(2**self._n_bits - 1, 1))
        elif self._encoding == "unary":
            n_ones = sum(qubits.get(i, 0) for i in range(self._n_bits))
            return n_ones / max(self._n_bits, 1)
        else:  # one_hot
            for i in range(self._n_bits):
                if qubits.get(i, 0) == 1:
                    return i / max(self._n_bits - 1, 1)
            return 0.0

    def qubits_needed(self, n_sc_values: int) -> int:
        """Total qubits needed to encode n SC values."""
        return n_sc_values * self._n_bits

    def encode_array(self, values: np.ndarray[Any, Any]) -> Dict[int, int]:
        """Encode array of SC values into a single qubit dict.

        Parameters
        ----------
        values : np.ndarray
            1D array of SC values.

        Returns
        -------
        dict[int, int]
            Global qubit index → binary value.
        """
        result: Dict[int, int] = {}
        for idx, v in enumerate(values):
            local = self.encode(float(v))
            for qi, val in local.items():
                result[idx * self._n_bits + qi] = val
        return result


# ══════════════════════════════════════════════════════════════════════
# Problem Decomposer (Qbsolv-style)
# ══════════════════════════════════════════════════════════════════════


class ProblemDecomposer:
    """Decompose large QUBO/Ising into sub-problems for QPU.

    When a model exceeds QPU capacity, this class partitions it
    into smaller sub-problems that fit on hardware, solves each,
    then merges the results.

    Parameters
    ----------
    max_subproblem_size : int
        Maximum qubits per sub-problem (default 64 for Chimera unit cell).
    overlap : int
        Number of shared qubits between partitions (default 4).
    n_iterations : int
        Number of decomposition-merge iterations (default 10).
    """

    def __init__(
        self,
        max_subproblem_size: int = 64,
        overlap: int = 4,
        n_iterations: int = 10,
    ) -> None:
        self._max_size = max_subproblem_size
        self._overlap = overlap
        self._n_iterations = n_iterations

    def decompose(self, model: IsingModel) -> list[IsingModel]:
        """Partition Ising model into sub-problems.

        Uses a greedy graph partitioning that keeps strongly-coupled
        qubits together.

        Parameters
        ----------
        model : IsingModel
            The model to decompose.

        Returns
        -------
        list[IsingModel]
            Sub-problems, each ≤ max_subproblem_size qubits.
        """
        if model.n_qubits <= self._max_size:
            return [model]

        # Build adjacency
        neighbors: Dict[int, list[int]] = {i: [] for i in range(model.n_qubits)}
        for i, j in model.J:
            neighbors[i].append(j)
            neighbors[j].append(i)

        # Greedy partitioning
        assigned: set[int] = set()
        partitions: list[list[int]] = []

        remaining = set(range(model.n_qubits))
        while remaining:
            seed = min(remaining)
            partition = [seed]
            assigned.add(seed)
            remaining.discard(seed)

            while len(partition) < self._max_size and remaining:
                # Find unassigned neighbor of current partition
                best = None
                best_score: float = -1.0
                for q in partition:
                    for n in neighbors.get(q, []):
                        if n in remaining:
                            score = abs(model.J.get((min(q, n), max(q, n)), 0.0))
                            if score > best_score:
                                best = n
                                best_score = score

                if best is None:
                    # No connected neighbors, take any remaining
                    best = min(remaining)

                partition.append(best)
                assigned.add(best)
                remaining.discard(best)

            partitions.append(partition)

        # Build sub-models
        sub_models: list[IsingModel] = []
        for part_idx, part_qubits in enumerate(partitions):
            qs = set(part_qubits)
            local_map = {q: i for i, q in enumerate(part_qubits)}

            h_sub = {local_map[q]: model.h.get(q, 0.0) for q in part_qubits}
            j_sub: Dict[tuple[int, int], float] = {}
            for (i, j), jij in model.J.items():
                if i in qs and j in qs:
                    li, lj = local_map[i], local_map[j]
                    a, b = min(li, lj), max(li, lj)
                    j_sub[(a, b)] = jij

            labels = {local_map[q]: model.qubit_labels.get(q, f"q{q}") for q in part_qubits}

            sub_models.append(
                IsingModel(
                    h=h_sub,
                    J=j_sub,
                    offset=0.0,
                    qubit_labels=labels,
                    n_qubits=len(part_qubits),
                    source=f"{model.source}_part{part_idx}",
                )
            )

        return sub_models

    def solve_decomposed(
        self,
        model: IsingModel,
        solver: SimulatedAnnealer | None = None,
    ) -> Dict[str, Any]:
        """Decompose, solve sub-problems, and merge.

        Parameters
        ----------
        model : IsingModel
            The full model.
        solver : SimulatedAnnealer | None
            Solver for sub-problems (default: new SA).

        Returns
        -------
        dict
            ``best_spins``, ``best_energy``, ``n_partitions``.
        """
        if solver is None:
            solver = SimulatedAnnealer(n_sweeps=1000, seed=42)

        sub_models = self.decompose(model)

        # Reconstruct global mapping
        global_spins: Dict[int, int] = {}
        # Initialize with +1
        for i in range(model.n_qubits):
            global_spins[i] = 1

        for _iteration in range(self._n_iterations):
            for sub in sub_models:
                result = solver.solve_ising(sub, num_reads=5)
                # Map back
                best = result["best_spins"]
                for local_q, spin in best.items():
                    # Find global index from label
                    label = sub.qubit_labels.get(local_q, "")
                    for gq, gl in model.qubit_labels.items():
                        if gl == label:
                            global_spins[gq] = spin
                            break

        return {
            "best_spins": global_spins,
            "best_energy": model.energy(global_spins),
            "n_partitions": len(sub_models),
            "n_iterations": self._n_iterations,
        }


# ══════════════════════════════════════════════════════════════════════
# Time-to-Solution (TTS) Analyzer
# ══════════════════════════════════════════════════════════════════════


class TTSAnalyzer:
    """Time-to-solution quality metric for quantum annealing.

    TTS measures the total time required to find the ground state
    with probability p_target, given:
    - p_success: probability of finding ground state in a single run
    - t_anneal: time per annealing run

    TTS = t_anneal × (log(1 - p_target) / log(1 - p_success))

    This is the standard benchmark metric used in D-Wave literature.
    """

    def compute(
        self,
        p_success: float,
        t_anneal_us: float,
        p_target: float = 0.99,
    ) -> Dict[str, float]:
        """Compute TTS metric.

        Parameters
        ----------
        p_success : float
            Probability of finding ground state per run.
        t_anneal_us : float
            Time per annealing run in microseconds.
        p_target : float
            Target cumulative success probability (default 0.99).

        Returns
        -------
        dict
            ``tts_us``, ``tts_ms``, ``n_runs_needed``,
            ``p_success``, ``p_target``.
        """
        if p_success <= 0:
            return {
                "tts_us": float("inf"),
                "tts_ms": float("inf"),
                "n_runs_needed": float("inf"),
                "p_success": 0.0,
                "p_target": p_target,
            }

        if p_success >= 1.0:
            return {
                "tts_us": t_anneal_us,
                "tts_ms": t_anneal_us / 1000.0,
                "n_runs_needed": 1.0,
                "p_success": 1.0,
                "p_target": p_target,
            }

        n_runs = math.log(1 - p_target) / math.log(1 - p_success)
        tts = t_anneal_us * n_runs

        return {
            "tts_us": tts,
            "tts_ms": tts / 1000.0,
            "n_runs_needed": n_runs,
            "p_success": p_success,
            "p_target": p_target,
        }

    def from_samples(
        self,
        energies: list[float],
        ground_state_energy: float,
        t_anneal_us: float = 20.0,
        tolerance: float = 1e-6,
        p_target: float = 0.99,
    ) -> Dict[str, float]:
        """Compute TTS from a set of sample energies.

        Parameters
        ----------
        energies : list[float]
            Observed sample energies.
        ground_state_energy : float
            Known or estimated ground state energy.
        t_anneal_us : float
            Time per annealing run.
        tolerance : float
            Energy tolerance for ground state match.
        p_target : float
            Target success probability.

        Returns
        -------
        dict
            TTS metrics.
        """
        n_gs = sum(1 for e in energies if abs(e - ground_state_energy) < tolerance)
        p_success = n_gs / max(len(energies), 1)
        return self.compute(p_success, t_anneal_us, p_target)

    def compare_solvers(
        self,
        results: Dict[str, Dict[str, Any]],
        ground_state_energy: float,
        tolerance: float = 1e-6,
    ) -> Dict[str, Dict[str, Any]]:
        """Compare TTS across multiple solvers.

        Parameters
        ----------
        results : dict
            Solver name → {energies, t_anneal_us}.
        ground_state_energy : float
            Known ground state energy.

        Returns
        -------
        dict
            Solver name → TTS metrics.
        """
        comparison: Dict[str, Dict[str, Any]] = {}
        for name, data in results.items():
            comparison[name] = self.from_samples(
                energies=data["energies"],
                ground_state_energy=ground_state_energy,
                t_anneal_us=data.get("t_anneal_us", 20.0),
                tolerance=tolerance,
            )
        return comparison
