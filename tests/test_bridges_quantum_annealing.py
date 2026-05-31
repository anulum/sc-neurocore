# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for bridges.quantum_annealing

from __future__ import annotations

import time

import numpy as np

import sc_neurocore.bridges.quantum_annealing as qa
from sc_neurocore.bridges.quantum_annealing import (
    IsingModel,
    QUBOModel,
    SCToIsing,
    SCToQUBO,
    SimulatedAnnealer,
    AnnealingSchedule,
    GaugeTransform,
    SCBitstreamQUBO,
    ProblemDecomposer,
    SCPrecisionEncoder,
)


# ---------------------------------------------------------------------------
# Ising Model — dict-based h,J
# ---------------------------------------------------------------------------


class TestIsingModel:
    def test_energy_ferromagnetic_ground_state(self):
        """FM chain: all spins aligned → minimal energy."""
        n = 5
        h = {i: 0.0 for i in range(n)}
        J = {}
        for i in range(n - 1):
            J[(i, i + 1)] = -1.0
        model = IsingModel(h=h, J=J)
        aligned = {i: 1 for i in range(n)}
        e_aligned = model.energy(aligned)
        alt = {i: (1 if i % 2 == 0 else -1) for i in range(n)}
        e_alt = model.energy(alt)
        assert e_aligned < e_alt

    def test_energy_with_field(self):
        h = {0: 1.0, 1: -1.0}
        J = {}
        model = IsingModel(h=h, J=J)
        e_pp = model.energy({0: 1, 1: 1})
        e_pm = model.energy({0: 1, 1: -1})
        assert e_pm != e_pp


# ---------------------------------------------------------------------------
# QUBO Model — dict-based Q
# ---------------------------------------------------------------------------


class TestQUBOModel:
    def test_energy_basic(self):
        Q = {(0, 0): -1.0, (1, 1): -1.0, (0, 1): 0.5}
        model = QUBOModel(Q=Q)
        e_00 = model.energy({0: 0, 1: 0})
        e_11 = model.energy({0: 1, 1: 1})
        assert e_11 < e_00

    def test_to_ising(self):
        Q = {(0, 0): -2.0, (1, 1): -2.0, (0, 1): 1.0}
        qubo = QUBOModel(Q=Q)
        ising = qubo.to_ising()
        assert isinstance(ising, qa.IsingModel)
        assert len(ising.h) == 2


# ---------------------------------------------------------------------------
# SC-to-Ising / SC-to-QUBO
# ---------------------------------------------------------------------------


class TestSCToIsing:
    def test_compile_simple_network(self):
        adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        compiler = SCToIsing()
        ising = compiler.compile(adj)
        assert isinstance(ising, qa.IsingModel)
        assert ising.n_qubits >= 3


class TestSCToQUBO:
    def test_compile_simple_network(self):
        adj = np.array([[0, 1], [1, 0]], dtype=float)
        compiler = SCToQUBO()
        qubo = compiler.compile(adj)
        assert isinstance(qubo, qa.QUBOModel)
        assert qubo.n_qubits >= 2


# ---------------------------------------------------------------------------
# Simulated Annealer
# ---------------------------------------------------------------------------


class TestSimulatedAnnealer:
    def _make_fm_chain(self, n=8):
        h = {i: 0.0 for i in range(n)}
        J = {(i, i + 1): -1.0 for i in range(n - 1)}
        return IsingModel(h=h, J=J)

    def test_solve_ising_fm_chain(self):
        model = self._make_fm_chain(8)
        sa = SimulatedAnnealer(n_sweeps=500, seed=42)
        result = sa.solve_ising(model)
        assert isinstance(result, dict)
        assert "sample" in result or "samples" in result

    def test_solve_qubo(self):
        Q = {(0, 0): -3.0, (1, 1): -3.0, (2, 2): -3.0, (0, 1): 1.0, (1, 2): 1.0}
        model = QUBOModel(Q=Q)
        sa = SimulatedAnnealer(n_sweeps=200, seed=42)
        result = sa.solve_qubo(model)
        assert result is not None

    def test_deterministic_with_seed(self):
        model = self._make_fm_chain(4)
        r1 = SimulatedAnnealer(n_sweeps=100, seed=7).solve_ising(model)
        r2 = SimulatedAnnealer(n_sweeps=100, seed=7).solve_ising(model)
        assert r1 == r2


# ---------------------------------------------------------------------------
# Annealing Schedule — instance methods
# ---------------------------------------------------------------------------


class TestAnnealingSchedule:
    def test_linear_schedule(self):
        sched = AnnealingSchedule()
        sched = sched.linear(duration_us=100.0)
        assert isinstance(sched, AnnealingSchedule)
        d = sched.to_dict()
        assert isinstance(d, dict)

    def test_pause_and_quench(self):
        sched = AnnealingSchedule()
        sched = sched.pause_and_quench()
        d = sched.to_dict()
        assert isinstance(d, dict)

    def test_reverse_schedule(self):
        sched = AnnealingSchedule()
        sched = sched.reverse()
        d = sched.to_dict()
        assert isinstance(d, dict)


# ---------------------------------------------------------------------------
# Gauge Transform
# ---------------------------------------------------------------------------


class TestGaugeTransform:
    def test_transform_returns_list(self):
        gt = GaugeTransform(n_gauges=3, seed=42)
        adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        compiler = SCToIsing()
        model = compiler.compile(adj)
        transformed = gt.transform(model)
        assert isinstance(transformed, list)
        assert len(transformed) >= 1


# ---------------------------------------------------------------------------
# SCBitstreamQUBO
# ---------------------------------------------------------------------------


class TestSCBitstreamQUBO:
    def test_weight_optimization(self):
        target = np.array([0.5, 0.3, 0.8])
        candidates = np.array([[0.4, 0.3, 0.7], [0.6, 0.2, 0.9], [0.5, 0.3, 0.8]])
        bq = SCBitstreamQUBO()
        result = bq.weight_optimization(target, candidates, n_bits=4)
        assert isinstance(result, qa.QUBOModel)

    def test_pruning(self):
        adj = np.array([[0, 0.01, 0.9], [0.01, 0, 0.8], [0.9, 0.8, 0]], dtype=float)
        importance = np.array([[0, 0.1, 0.9], [0.1, 0, 0.8], [0.9, 0.8, 0]], dtype=float)
        bq = SCBitstreamQUBO()
        result = bq.pruning(adj, importance, max_connections=2)
        assert isinstance(result, qa.QUBOModel)


# ---------------------------------------------------------------------------
# Problem Decomposer
# ---------------------------------------------------------------------------


class TestProblemDecomposer:
    def _make_random_ising(self, n=12, seed=42):
        rng = np.random.default_rng(seed)
        h = {i: float(rng.standard_normal()) for i in range(n)}
        J = {}
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() > 0.5:
                    J[(i, j)] = float(rng.standard_normal())
        return IsingModel(h=h, J=J)

    def test_decompose_returns_list(self):
        decomposer = ProblemDecomposer(max_subproblem_size=5)
        model = self._make_random_ising(12)
        subs = decomposer.decompose(model)
        assert isinstance(subs, list)
        assert len(subs) >= 1

    def test_solve_decomposed(self):
        decomposer = ProblemDecomposer(max_subproblem_size=5, n_iterations=2)
        model = self._make_random_ising(10)
        result = decomposer.solve_decomposed(model)
        assert result is not None


# ---------------------------------------------------------------------------
# SCPrecisionEncoder
# ---------------------------------------------------------------------------


class TestSCPrecisionEncoder:
    def test_encode_decode_roundtrip(self):
        encoder = SCPrecisionEncoder(n_bits=8)
        val = 0.75
        encoded = encoder.encode(val)
        decoded = encoder.decode(encoded)
        assert abs(decoded - val) < 1.0 / encoder.n_levels

    def test_encode_array(self):
        encoder = SCPrecisionEncoder(n_bits=4)
        arr = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        encoded = encoder.encode_array(arr)
        assert isinstance(encoded, dict)
        assert len(encoded) == 5 * 4  # n_bits qubits per value

    def test_qubits_needed(self):
        encoder = SCPrecisionEncoder(n_bits=8)
        q = encoder.qubits_needed(10)
        assert q == 80


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


class TestQuantumAnnealingBenchmark:
    def test_solve_64_spin_ising(self):
        """SA must solve 64-spin random Ising in < 10 seconds."""
        rng = np.random.default_rng(42)
        n = 64
        h = {i: float(rng.standard_normal()) for i in range(n)}
        J = {}
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() > 0.7:
                    J[(i, j)] = float(rng.standard_normal())
        model = IsingModel(h=h, J=J)
        sa = SimulatedAnnealer(n_sweeps=500, seed=42)
        t0 = time.perf_counter()
        result = sa.solve_ising(model)
        elapsed = time.perf_counter() - t0
        assert result is not None
        assert elapsed < 10.0, f"64-spin SA took {elapsed:.2f}s"

    def test_compile_50_node_sc_to_ising(self):
        """Compile 50-node SC network to Ising in < 2 seconds."""
        rng = np.random.default_rng(42)
        adj = (rng.random((50, 50)) > 0.8).astype(float)
        np.fill_diagonal(adj, 0)
        compiler = SCToIsing()
        t0 = time.perf_counter()
        ising = compiler.compile(adj)
        elapsed = time.perf_counter() - t0
        assert ising.n_qubits >= 50
        assert elapsed < 2.0, f"50-node SC→Ising took {elapsed:.2f}s"
