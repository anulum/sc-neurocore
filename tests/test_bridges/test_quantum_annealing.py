# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Tests for sc_neurocore.bridges.quantum_annealing."""

from __future__ import annotations

import builtins
import importlib
import json
import os
import sys
import types

import numpy as np
import pytest

import sc_neurocore.bridges.quantum_annealing as qa
from sc_neurocore.bridges.quantum_annealing import (
    AnnealingSchedule,
    ChainBreakResolver,
    CouplerSpec,
    DWaveInterface,
    EmbeddingAnalyzer,
    EnergyLandscape,
    GaugeTransform,
    HardwareGraph,
    IsingModel,
    ProblemType,
    QUBOModel,
    QubitSpec,
    SampleAggregator,
    SCBitstreamQUBO,
    SCPrecisionEncoder,
    SCToIsing,
    SCToQUBO,
    SimulatedAnnealer,
    ProblemDecomposer,
    TTSAnalyzer,
    export_bqm,
    export_ising_json,
    export_qubo_json,
    visualize_ising,
)


# ══════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture
def simple_adjacency() -> np.ndarray:
    """3-node excitatory network."""
    return np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )


@pytest.fixture
def simple_ising() -> IsingModel:
    """Minimal 3-qubit Ising model."""
    return IsingModel(
        h={0: 0.1, 1: -0.2, 2: 0.0},
        J={(0, 1): -1.0, (1, 2): 0.5},
        offset=0.0,
        qubit_labels={0: "A", 1: "B", 2: "C"},
        n_qubits=3,
        source="test",
    )


# ══════════════════════════════════════════════════════════════════════
# 1. Data Types
# ══════════════════════════════════════════════════════════════════════


class TestDataTypes:
    """Ising/QUBO data class tests."""

    def test_optional_dependency_import_fallbacks_preserve_unavailable_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_import = builtins.__import__

        def guarded_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "dimod" or name == "dwave.system" or name.startswith("dwave."):
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        try:
            monkeypatch.setattr(builtins, "__import__", guarded_import)
            module = importlib.reload(qa)

            assert module._HAS_DIMOD is False
            assert module.dimod is None
            assert module._HAS_DWAVE is False
            assert module.DWaveSampler is None
            assert module.EmbeddingComposite is None
        finally:
            monkeypatch.setattr(builtins, "__import__", real_import)
            importlib.reload(qa)

    def test_optional_dependency_import_success_sets_available_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_dimod = types.ModuleType("dimod")
        fake_dwave = types.ModuleType("dwave")
        fake_dwave_system = types.ModuleType("dwave.system")

        class FakeSampler:
            pass

        class FakeEmbeddingComposite:
            pass

        fake_dwave_system.DWaveSampler = FakeSampler
        fake_dwave_system.EmbeddingComposite = FakeEmbeddingComposite
        fake_dwave.system = fake_dwave_system

        try:
            monkeypatch.setitem(sys.modules, "dimod", fake_dimod)
            monkeypatch.setitem(sys.modules, "dwave", fake_dwave)
            monkeypatch.setitem(sys.modules, "dwave.system", fake_dwave_system)
            module = importlib.reload(qa)

            assert module._HAS_DIMOD is True
            assert module.dimod is fake_dimod
            assert module._HAS_DWAVE is True
            assert module.DWaveSampler is FakeSampler
            assert module.EmbeddingComposite is FakeEmbeddingComposite
        finally:
            sys.modules.pop("dimod", None)
            sys.modules.pop("dwave", None)
            sys.modules.pop("dwave.system", None)
            importlib.reload(qa)

    def test_qubit_spec(self) -> None:
        q = QubitSpec(index=0, label="neuron_0", bias=0.5)
        assert q.index == 0
        assert q.bias == 0.5

    def test_coupler_spec(self) -> None:
        c = CouplerSpec(qubit_a=0, qubit_b=1, strength=-1.0)
        assert c.strength == -1.0

    def test_problem_type_enum(self) -> None:
        assert ProblemType.ISING.value == "ising"
        assert ProblemType.QUBO.value == "qubo"

    def test_ising_energy(self, simple_ising: IsingModel) -> None:
        spins = {0: 1, 1: 1, 2: -1}
        e = simple_ising.energy(spins)
        expected = 0.1 * 1 + (-0.2) * 1 + 0.0 * (-1) + (-1.0) * 1 * 1 + 0.5 * 1 * (-1)
        assert abs(e - expected) < 1e-10

    def test_qubo_energy(self) -> None:
        qubo = QUBOModel(
            Q={(0, 0): -1.0, (1, 1): -1.0, (0, 1): 2.0},
            n_qubits=2,
        )
        e = qubo.energy({0: 1, 1: 1})
        assert abs(e - 0.0) < 1e-10  # -1 -1 + 2 = 0

    def test_qubo_to_ising(self) -> None:
        qubo = QUBOModel(
            Q={(0, 0): -1.0, (1, 1): -1.0, (0, 1): 2.0},
            n_qubits=2,
            source="test_qubo",
        )
        ising = qubo.to_ising()
        assert ising.n_qubits == 2
        assert "QUBO→Ising" in ising.source

    def test_large_ising_delegates_to_native_energy_contract(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        def fake_energy(
            h_indices: list[int],
            h_values: list[float],
            j_i: list[int],
            j_j: list[int],
            j_values: list[float],
            spin_arr: list[int],
            offset: float,
        ) -> float:
            captured["h_indices"] = h_indices
            captured["h_values"] = h_values
            captured["j_i"] = j_i
            captured["j_j"] = j_j
            captured["j_values"] = j_values
            captured["spin_arr"] = spin_arr
            captured["offset"] = offset
            return -7.5

        monkeypatch.setattr(qa, "_HAS_RUST_QA", True)
        monkeypatch.setattr(qa, "_rust_ising_energy", fake_energy)

        model = IsingModel(
            h={0: 0.25, 20: -0.5},
            J={(0, 20): -1.25},
            offset=1.0,
            n_qubits=21,
            source="native_contract",
        )
        energy = model.energy({0: -1, 20: 1})

        assert energy == -7.5
        assert captured == {
            "h_indices": [0, 20],
            "h_values": [0.25, -0.5],
            "j_i": [0],
            "j_j": [20],
            "j_values": [-1.25],
            "spin_arr": [-1] + [1] * 20,
            "offset": 1.0,
        }


# ══════════════════════════════════════════════════════════════════════
# 2. SC-to-Ising Compiler
# ══════════════════════════════════════════════════════════════════════


class TestSCToIsing:
    """SC network to Ising compilation."""

    def test_compile_basic(self, simple_adjacency: np.ndarray) -> None:
        compiler = SCToIsing()
        model = compiler.compile(simple_adjacency, name="test")
        assert model.n_qubits == 3
        assert len(model.J) > 0
        assert model.source == "test"

    def test_excitatory_ferromagnetic(self, simple_adjacency: np.ndarray) -> None:
        compiler = SCToIsing()
        model = compiler.compile(simple_adjacency)
        # Excitatory → J < 0 (ferromagnetic)
        for jij in model.J.values():
            assert jij < 0

    def test_inhibitory_antiferromagnetic(self) -> None:
        adj = np.array([[0.0, -1.0], [-1.0, 0.0]])
        compiler = SCToIsing()
        model = compiler.compile(adj)
        for jij in model.J.values():
            assert jij > 0  # antiferromagnetic

    def test_custom_labels(self, simple_adjacency: np.ndarray) -> None:
        compiler = SCToIsing()
        model = compiler.compile(simple_adjacency, node_labels=["X", "Y", "Z"])
        assert model.qubit_labels[0] == "X"

    def test_with_biases(self, simple_adjacency: np.ndarray) -> None:
        compiler = SCToIsing(field_scale=1.0)
        biases = np.array([0.5, -0.3, 0.0])
        model = compiler.compile(simple_adjacency, biases=biases)
        assert abs(model.h[0] - 0.5) < 1e-10


# ══════════════════════════════════════════════════════════════════════
# 3. SC-to-QUBO Compiler
# ══════════════════════════════════════════════════════════════════════


class TestSCToQUBO:
    """SC network to QUBO compilation."""

    def test_compile_basic(self, simple_adjacency: np.ndarray) -> None:
        compiler = SCToQUBO()
        model = compiler.compile(simple_adjacency)
        assert model.n_qubits == 3
        assert len(model.Q) > 0

    def test_diagonal_present(self, simple_adjacency: np.ndarray) -> None:
        compiler = SCToQUBO()
        model = compiler.compile(simple_adjacency)
        assert (0, 0) in model.Q


# ══════════════════════════════════════════════════════════════════════
# 4. Simulated Annealing
# ══════════════════════════════════════════════════════════════════════


class TestSimulatedAnnealer:
    """Simulated annealing solver."""

    def test_solve_ising(self, simple_ising: IsingModel) -> None:
        sa = SimulatedAnnealer(n_sweeps=100, seed=42)
        result = sa.solve_ising(simple_ising, num_reads=5)
        assert "best_spins" in result
        assert "best_energy" in result
        assert len(result["samples"]) == 5

    def test_spins_valid(self, simple_ising: IsingModel) -> None:
        sa = SimulatedAnnealer(n_sweeps=100, seed=42)
        result = sa.solve_ising(simple_ising, num_reads=3)
        for spin in result["best_spins"].values():
            assert spin in (-1, 1)

    def test_solve_qubo(self) -> None:
        qubo = QUBOModel(
            Q={(0, 0): -1.0, (1, 1): -1.0, (0, 1): 2.0},
            n_qubits=2,
        )
        sa = SimulatedAnnealer(n_sweeps=200, seed=42)
        result = sa.solve_qubo(qubo, num_reads=5)
        assert "best_bits" in result
        for bit in result["best_bits"].values():
            assert bit in (0, 1)

    def test_finds_ground_state_2qubit(self) -> None:
        # Simple ferromagnetic: J < 0 → ground state is aligned
        model = IsingModel(
            h={0: 0.0, 1: 0.0},
            J={(0, 1): -1.0},
            n_qubits=2,
        )
        sa = SimulatedAnnealer(n_sweeps=2000, seed=42)
        result = sa.solve_ising(model, num_reads=20)
        # Ground state energy = -1.0 (both aligned)
        assert result["best_energy"] <= -0.99

    def test_native_solver_result_preserves_sample_contract(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        def fake_solver(*args: object) -> dict[str, object]:
            captured["args"] = args
            return {
                "best_spins": [1, -1, 1] + [1] * 9,
                "best_energy": -3.0,
                "energies": [-3.0, -2.0],
                "samples": [[1, -1, 1] + [1] * 9, [-1, -1, 1] + [1] * 9],
            }

        monkeypatch.setattr(qa, "_HAS_RUST_QA", True)
        monkeypatch.setattr(qa, "_rust_sa", fake_solver)

        model = IsingModel(
            h={0: 0.5, 2: -0.25},
            J={(0, 1): -1.0, (1, 2): 0.75},
            offset=0.5,
            n_qubits=12,
            source="native_sa_contract",
        )
        result = SimulatedAnnealer(n_sweeps=17, beta_start=0.2, beta_end=3.0).solve_ising(
            model, num_reads=2
        )

        assert result["backend"] == "rust"
        assert result["best_spins"][0] == 1
        assert result["best_spins"][1] == -1
        assert result["samples"][1][0] == -1
        assert result["energies"] == [-3.0, -2.0]
        assert captured["args"] == (
            [0, 2],
            [0.5, -0.25],
            [0, 1],
            [1, 2],
            [-1.0, 0.75],
            12,
            0.5,
            17,
            2,
            0.2,
            3.0,
            42,
        )


# ══════════════════════════════════════════════════════════════════════
# 5. D-Wave Interface
# ══════════════════════════════════════════════════════════════════════


class TestDWaveInterface:
    """D-Wave interface (fallback to SA)."""

    def test_availability(self) -> None:
        dw = DWaveInterface()
        # Should return bool regardless
        assert isinstance(dw.available, bool)

    def test_fallback_solve(self, simple_ising: IsingModel) -> None:
        import os

        dw = DWaveInterface(num_reads=5)
        # If D-Wave SDK is installed, solve_ising() hits the real sampler
        # and needs an API token — skip unless one is configured.
        if dw.available and not os.environ.get("DWAVE_API_TOKEN"):
            pytest.skip(
                "D-Wave SDK installed but DWAVE_API_TOKEN not set; cannot exercise QPU path"
            )
        result = dw.solve_ising(simple_ising)
        assert "best_spins" in result
        assert result.get("backend") in ("simulated_annealing_fallback", "dwave_qpu")

    def test_qpu_path_submits_bqm_and_reports_timing(
        self, simple_ising: IsingModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        class FakeDimod:
            class BinaryQuadraticModel:
                def __init__(
                    self,
                    h: dict[int, float],
                    j_couplings: dict[tuple[int, int], float],
                    offset: float,
                    vartype: str,
                ) -> None:
                    captured["bqm"] = {
                        "h": h,
                        "J": j_couplings,
                        "offset": offset,
                        "vartype": vartype,
                    }

        class FakeSampler:
            pass

        class FakeBest:
            sample = {0: 1, 1: -1, 2: 1}
            energy = -1.25

        class FakeResponse:
            first = FakeBest()
            info = {"timing": {"qpu_access_time": 123}}

        class FakeEmbeddingComposite:
            def __init__(self, sampler: FakeSampler) -> None:
                captured["sampler"] = sampler

            def sample(self, bqm: object, **kwargs: object) -> FakeResponse:
                captured["sample_kwargs"] = kwargs
                captured["sample_bqm"] = bqm
                return FakeResponse()

        monkeypatch.setattr(qa, "_HAS_DWAVE", True)
        monkeypatch.setattr(qa, "_HAS_DIMOD", True)
        monkeypatch.setattr(qa, "dimod", FakeDimod)
        monkeypatch.setattr(qa, "DWaveSampler", FakeSampler)
        monkeypatch.setattr(qa, "EmbeddingComposite", FakeEmbeddingComposite)

        result = DWaveInterface(
            chain_strength=1.7, num_reads=31, annealing_time_us=23.0
        ).solve_ising(simple_ising)

        assert result == {
            "best_spins": {0: 1, 1: -1, 2: 1},
            "best_energy": -1.25,
            "num_reads": 31,
            "backend": "dwave_qpu",
            "timing": {"qpu_access_time": 123},
        }
        assert captured["bqm"] == {
            "h": simple_ising.h,
            "J": simple_ising.J,
            "offset": simple_ising.offset,
            "vartype": "SPIN",
        }
        assert captured["sample_kwargs"] == {
            "num_reads": 31,
            "chain_strength": 1.7,
            "annealing_time": 23.0,
        }


# ══════════════════════════════════════════════════════════════════════
# 6. Energy Landscape
# ══════════════════════════════════════════════════════════════════════


class TestEnergyLandscape:
    """Energy landscape analysis."""

    def test_analyze_small(self, simple_ising: IsingModel) -> None:
        el = EnergyLandscape()
        result = el.analyze(simple_ising)
        assert "min_energy" in result
        assert "spectral_gap" in result
        assert "degeneracy" in result
        assert result["n_samples"] == 2**3  # full enumeration for n=3

    def test_min_leq_max(self, simple_ising: IsingModel) -> None:
        result = EnergyLandscape().analyze(simple_ising)
        assert result["min_energy"] <= result["max_energy"]

    def test_with_precomputed_samples(self, simple_ising: IsingModel) -> None:
        samples = [{0: 1, 1: 1, 2: 1}, {0: -1, 1: -1, 2: -1}]
        result = EnergyLandscape().analyze(simple_ising, samples=samples)
        assert result["n_samples"] == 2

    def test_large_python_sampling_reports_finite_landscape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(qa, "_HAS_RUST_QA", False)
        model = IsingModel(h={0: -1.0}, J={}, n_qubits=21, source="large_python")

        result = EnergyLandscape().analyze(model)

        assert result["n_samples"] == 10000
        assert result["min_energy"] == -1.0
        assert result["max_energy"] == 1.0
        assert result["spectral_gap"] == 2.0
        assert result["degeneracy"] > 0

    def test_native_vector_energy_contract(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_vector_energy(*args: object) -> list[float]:
            captured["args"] = args
            return [-1.0, -1.0, 0.5] + [1.0] * 98

        monkeypatch.setattr(qa, "_HAS_RUST_QA", True)
        monkeypatch.setattr(qa, "_rust_batch_energy", fake_vector_energy)
        samples = [{0: 1 if idx % 2 == 0 else -1, 1: -1, 2: 1} for idx in range(101)]
        model = IsingModel(
            h={0: 0.5},
            J={(1, 2): -0.25},
            offset=0.0,
            n_qubits=3,
            source="native_vector_contract",
        )

        result = EnergyLandscape().analyze(model, samples=samples)

        assert result["min_energy"] == -1.0
        assert result["degeneracy"] == 2
        assert result["spectral_gap"] == 1.5
        assert result["n_unique_energies"] == 3
        assert captured["args"] == (
            [0],
            [0.5],
            [1],
            [2],
            [-0.25],
            [[s.get(i, 1) for i in range(3)] for s in samples],
            0.0,
        )


# ══════════════════════════════════════════════════════════════════════
# 7. Embedding Analyzer
# ══════════════════════════════════════════════════════════════════════


class TestEmbeddingAnalyzer:
    """Embedding requirement analysis."""

    def test_analyze(self, simple_ising: IsingModel) -> None:
        ea = EmbeddingAnalyzer()
        result = ea.analyze(simple_ising)
        assert result["n_logical_qubits"] == 3
        assert result["n_couplers"] == 2
        assert isinstance(result["pegasus_compatible"], bool)

    def test_dense_graph(self) -> None:
        # Fully connected 5-node
        adj = np.ones((5, 5)) - np.eye(5)
        model = SCToIsing().compile(adj)
        result = EmbeddingAnalyzer().analyze(model)
        assert result["density"] > 0.5


# ══════════════════════════════════════════════════════════════════════
# 8. Export Functions
# ══════════════════════════════════════════════════════════════════════


class TestExportFunctions:
    """JSON export tests."""

    def test_export_ising_json(self, simple_ising: IsingModel, tmp_path: str) -> None:
        path = os.path.join(tmp_path, "test_ising.json")
        export_ising_json(simple_ising, path)
        with open(path) as f:
            data = json.load(f)
        assert data["type"] == "ising"
        assert data["n_qubits"] == 3

    def test_export_qubo_json(self, simple_adjacency: np.ndarray, tmp_path: str) -> None:
        qubo = SCToQUBO().compile(simple_adjacency)
        path = os.path.join(tmp_path, "test_qubo.json")
        export_qubo_json(qubo, path)
        with open(path) as f:
            data = json.load(f)
        assert data["type"] == "qubo"

    def test_export_bqm_without_dimod(self, simple_ising: IsingModel) -> None:
        result = export_bqm(simple_ising)
        # Returns None if dimod not installed, BQM otherwise
        assert result is None or hasattr(result, "to_ising")

    def test_export_bqm_preserves_spin_model_contract(
        self, simple_ising: IsingModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        class FakeDimod:
            class BinaryQuadraticModel:
                def __init__(
                    self,
                    h: dict[int, float],
                    j_couplings: dict[tuple[int, int], float],
                    offset: float,
                    vartype: str,
                ) -> None:
                    captured["h"] = h
                    captured["J"] = j_couplings
                    captured["offset"] = offset
                    captured["vartype"] = vartype

        monkeypatch.setattr(qa, "_HAS_DIMOD", True)
        monkeypatch.setattr(qa, "dimod", FakeDimod)

        bqm = export_bqm(simple_ising)

        assert bqm is not None
        assert captured == {
            "h": simple_ising.h,
            "J": simple_ising.J,
            "offset": simple_ising.offset,
            "vartype": "SPIN",
        }


# ══════════════════════════════════════════════════════════════════════
# 9. Visualization
# ══════════════════════════════════════════════════════════════════════


class TestVisualization:
    """ASCII visualization."""

    def test_visualize_ising(self, simple_ising: IsingModel) -> None:
        viz = visualize_ising(simple_ising)
        assert "Ising Model" in viz
        assert "Biases" in viz
        assert "Couplings" in viz
        assert "ferro" in viz or "anti" in viz


# ══════════════════════════════════════════════════════════════════════
# 10. Hardware Graph
# ══════════════════════════════════════════════════════════════════════


class TestHardwareGraph:
    """D-Wave hardware topology model."""

    def test_pegasus_qubits(self) -> None:
        hw = HardwareGraph(topology="pegasus", size=16)
        assert hw.n_physical_qubits == 24 * 16 * 15  # 5760
        assert hw.connectivity == 15

    def test_chimera_qubits(self) -> None:
        hw = HardwareGraph(topology="chimera", size=16)
        assert hw.n_physical_qubits == 16 * 16 * 8  # 2048

    def test_zephyr_qubits(self) -> None:
        hw = HardwareGraph(topology="zephyr", size=12)
        assert hw.n_physical_qubits == 48 * 12 * 12

    def test_can_embed_small(self, simple_ising: IsingModel) -> None:
        hw = HardwareGraph(topology="pegasus", size=16)
        result = hw.can_embed(simple_ising)
        assert result["embeddable"] is True
        assert result["utilization_pct"] < 1.0

    def test_invalid_topology(self) -> None:
        with pytest.raises(ValueError):
            HardwareGraph(topology="flux_qubit")


# ══════════════════════════════════════════════════════════════════════
# 11. Chain Break Resolver
# ══════════════════════════════════════════════════════════════════════


class TestChainBreakResolver:
    """Broken chain repair."""

    def test_majority_vote(self) -> None:
        chains = {0: [0, 1, 2], 1: [3, 4]}
        samples = [{0: 1, 1: 1, 2: -1, 3: -1, 4: -1}]
        resolver = ChainBreakResolver(method="majority_vote")
        resolved = resolver.resolve(samples, chains)
        assert resolved[0][0] == 1  # 2 vs 1 → +1
        assert resolved[0][1] == -1  # 2 vs 0 → -1

    def test_minimize_energy(self, simple_ising: IsingModel) -> None:
        chains = {0: [0], 1: [1], 2: [2]}
        samples = [{0: 1, 1: -1, 2: 1}]
        resolver = ChainBreakResolver(method="minimize_energy")
        resolved = resolver.resolve(samples, chains, model=simple_ising)
        assert len(resolved) == 1

    def test_analyze_breaks(self) -> None:
        chains = {0: [0, 1], 1: [2, 3]}
        samples = [
            {0: 1, 1: -1, 2: 1, 3: 1},  # chain 0 broken
            {0: 1, 1: 1, 2: -1, 3: 1},  # chain 1 broken
        ]
        result = ChainBreakResolver().analyze_breaks(samples, chains)
        assert result["total_breaks"] == 2
        assert result["n_chains"] == 2

    def test_single_physical_qubit_chain_has_zero_break_rate(self) -> None:
        chains = {0: [10], 1: [20, 21]}
        samples = [{10: -1, 20: 1, 21: -1}]

        result = ChainBreakResolver().analyze_breaks(samples, chains)

        assert result["per_chain"][0] == 0.0
        assert result["per_chain"][1] == 1.0
        assert result["break_rate"] == 1.0

    def test_invalid_method(self) -> None:
        with pytest.raises(ValueError):
            ChainBreakResolver(method="random")


# ══════════════════════════════════════════════════════════════════════
# 12. Annealing Schedule
# ══════════════════════════════════════════════════════════════════════


class TestAnnealingSchedule:
    """Custom annealing schedule builder."""

    def test_linear(self) -> None:
        sched = AnnealingSchedule().linear(20.0)
        assert len(sched.points) == 2
        assert sched.points[0] == (0.0, 0.0)
        assert sched.points[-1] == (20.0, 1.0)
        assert sched.total_time_us == 20.0

    def test_pause_and_quench(self) -> None:
        sched = AnnealingSchedule().pause_and_quench(
            ramp_time_us=5.0, pause_at_s=0.4, pause_duration_us=50.0
        )
        assert len(sched.points) == 4
        assert sched.points[1][1] == 0.4  # paused at s=0.4
        assert sched.points[-1][1] == 1.0  # quenched to s=1

    def test_reverse(self) -> None:
        sched = AnnealingSchedule().reverse(reverse_to_s=0.3)
        assert sched.points[0][1] == 1.0  # starts at s=1
        assert sched.points[1][1] == 0.3  # reverses to 0.3
        assert sched.points[-1][1] == 1.0  # returns to s=1

    def test_to_dict(self) -> None:
        d = AnnealingSchedule().linear(20.0).to_dict()
        assert d["total_time_us"] == 20.0
        assert d["n_points"] == 2


# ══════════════════════════════════════════════════════════════════════
# 13. Gauge Transform
# ══════════════════════════════════════════════════════════════════════


class TestGaugeTransform:
    """Random gauge transformations."""

    def test_transform_count(self, simple_ising: IsingModel) -> None:
        gt = GaugeTransform(n_gauges=5, seed=42)
        transforms = gt.transform(simple_ising)
        assert len(transforms) == 5

    def test_preserves_structure(self, simple_ising: IsingModel) -> None:
        gt = GaugeTransform(n_gauges=1, seed=42)
        transformed = gt.transform(simple_ising)[0]
        assert transformed.n_qubits == simple_ising.n_qubits
        assert len(transformed.J) == len(simple_ising.J)

    def test_untransform_roundtrip(self) -> None:
        gt = GaugeTransform()
        gauge = {0: 1, 1: -1, 2: 1}
        sample = {0: 1, 1: -1, 2: 1}
        transformed = {i: s * gauge[i] for i, s in sample.items()}
        recovered = gt.untransform_sample(transformed, gauge)
        assert recovered == sample


# ══════════════════════════════════════════════════════════════════════
# 14. SC Bitstream QUBO
# ══════════════════════════════════════════════════════════════════════


class TestSCBitstreamQUBO:
    """SC-specific QUBO formulations."""

    def test_weight_optimization(self) -> None:
        W = np.array([[1.0, 0.5], [0.3, 0.8], [0.2, 0.6]])
        y = np.array([1.0, 0.5, 0.3])
        qubo = SCBitstreamQUBO().weight_optimization(y, W, n_bits=2)
        assert qubo.n_qubits == 2
        assert qubo.source == "sc_weight_optimization"

    def test_weight_qubo_solvable(self) -> None:
        W = np.eye(3)
        y = np.array([1.0, 0.0, 1.0])
        qubo = SCBitstreamQUBO().weight_optimization(y, W, n_bits=3)
        # Optimal x=[1,0,1] should have lower energy than x=[0,0,0]
        e_opt = qubo.energy({0: 1, 1: 0, 2: 1})
        e_zero = qubo.energy({0: 0, 1: 0, 2: 0})
        assert e_opt < e_zero

    def test_pruning(self) -> None:
        adj = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        importance = np.array([[0.0, 0.9, 0.1], [0.9, 0.0, 0.5], [0.1, 0.5, 0.0]])
        qubo = SCBitstreamQUBO().pruning(adj, importance, max_connections=2)
        assert qubo.n_qubits == 3  # 3 edges in a 3-node full graph
        assert qubo.source == "sc_pruning"


# ══════════════════════════════════════════════════════════════════════
# 15. Sample Aggregator
# ══════════════════════════════════════════════════════════════════════


class TestSampleAggregator:
    """Post-processing and sample aggregation."""

    def test_aggregate_basic(self) -> None:
        samples = [{0: 1, 1: 1}, {0: -1, 1: -1}, {0: 1, 1: 1}]
        energies = [-1.0, -1.0, 0.5]
        result = SampleAggregator().aggregate(samples, energies)
        assert result["unique_samples"] == 2
        assert result["total_samples"] == 3
        assert result["best_energy"] == -1.0

    def test_success_probability(self) -> None:
        samples = [{0: 1}] * 10
        energies = [-1.0] * 7 + [0.0] * 3
        result = SampleAggregator().aggregate(samples, energies)
        assert abs(result["success_probability"] - 0.7) < 1e-10

    def test_empty_samples(self) -> None:
        result = SampleAggregator().aggregate([], [])
        assert result["unique_samples"] == 0

    def test_boltzmann_weighting(self) -> None:
        samples = [{0: 1}, {0: -1}]
        energies = [0.0, 10.0]
        result = SampleAggregator().aggregate(samples, energies, temperature=1.0)
        # Low-energy sample should dominate
        assert result["boltzmann_avg_energy"] < 5.0


# ══════════════════════════════════════════════════════════════════════
# 16. SC Precision Encoder
# ══════════════════════════════════════════════════════════════════════


class TestSCPrecisionEncoder:
    """SC probability → qubit encoding."""

    def test_binary_roundtrip(self) -> None:
        enc = SCPrecisionEncoder(encoding="binary", n_bits=8)
        for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
            qubits = enc.encode(v)
            decoded = enc.decode(qubits)
            assert abs(decoded - v) < 0.01

    def test_unary_roundtrip(self) -> None:
        enc = SCPrecisionEncoder(encoding="unary", n_bits=8)
        qubits = enc.encode(0.5)
        decoded = enc.decode(qubits)
        assert abs(decoded - 0.5) < 0.01

    def test_one_hot_roundtrip(self) -> None:
        enc = SCPrecisionEncoder(encoding="one_hot", n_bits=8)
        qubits = enc.encode(0.0)
        decoded = enc.decode(qubits)
        assert abs(decoded) < 0.01

    def test_one_hot_empty_vector_decodes_to_zero(self) -> None:
        enc = SCPrecisionEncoder(encoding="one_hot", n_bits=4)
        assert enc.decode({}) == 0.0

    def test_n_levels(self) -> None:
        assert SCPrecisionEncoder("binary", 8).n_levels == 256
        assert SCPrecisionEncoder("unary", 8).n_levels == 9
        assert SCPrecisionEncoder("one_hot", 8).n_levels == 8

    def test_qubits_needed(self) -> None:
        enc = SCPrecisionEncoder("binary", 8)
        assert enc.qubits_needed(10) == 80

    def test_encode_array(self) -> None:
        enc = SCPrecisionEncoder("binary", 4)
        result = enc.encode_array(np.array([0.0, 1.0]))
        assert len(result) == 8  # 2 values × 4 bits

    def test_invalid_encoding(self) -> None:
        with pytest.raises(ValueError):
            SCPrecisionEncoder(encoding="gray")


# ══════════════════════════════════════════════════════════════════════
# 17. Problem Decomposer
# ══════════════════════════════════════════════════════════════════════


class TestProblemDecomposer:
    """Qbsolv-style problem decomposition."""

    def test_small_model_no_split(self, simple_ising: IsingModel) -> None:
        pd = ProblemDecomposer(max_subproblem_size=100)
        subs = pd.decompose(simple_ising)
        assert len(subs) == 1  # 3 qubits < 100

    def test_forced_split(self) -> None:
        adj = np.ones((10, 10)) - np.eye(10)
        model = SCToIsing().compile(adj, node_labels=[f"n{i}" for i in range(10)])
        pd = ProblemDecomposer(max_subproblem_size=4)
        subs = pd.decompose(model)
        assert len(subs) >= 3  # 10 qubits / 4 max = at least 3
        for sub in subs:
            assert sub.n_qubits <= 4

    def test_decompose_disconnected_model_preserves_all_qubits(self) -> None:
        model = IsingModel(
            h={i: float(i) for i in range(5)},
            J={},
            qubit_labels={i: f"q{i}" for i in range(5)},
            n_qubits=5,
            source="disconnected",
        )

        subs = ProblemDecomposer(max_subproblem_size=2).decompose(model)

        assert [sub.n_qubits for sub in subs] == [2, 2, 1]
        assert [label for sub in subs for label in sub.qubit_labels.values()] == [
            "q0",
            "q1",
            "q2",
            "q3",
            "q4",
        ]

    def test_solve_decomposed(self, simple_ising: IsingModel) -> None:
        pd = ProblemDecomposer(max_subproblem_size=100, n_iterations=2)
        result = pd.solve_decomposed(simple_ising)
        assert "best_spins" in result
        assert "best_energy" in result


# ══════════════════════════════════════════════════════════════════════
# 18. TTS Analyzer
# ══════════════════════════════════════════════════════════════════════


class TestTTSAnalyzer:
    """Time-to-solution quality metric."""

    def test_compute_basic(self) -> None:
        tts = TTSAnalyzer()
        result = tts.compute(p_success=0.5, t_anneal_us=20.0)
        assert result["n_runs_needed"] > 1
        assert result["tts_us"] > 20.0

    def test_perfect_success(self) -> None:
        result = TTSAnalyzer().compute(p_success=1.0, t_anneal_us=20.0)
        assert result["n_runs_needed"] == 1.0
        assert result["tts_us"] == 20.0

    def test_zero_success(self) -> None:
        result = TTSAnalyzer().compute(p_success=0.0, t_anneal_us=20.0)
        assert result["tts_us"] == float("inf")

    def test_from_samples(self) -> None:
        energies = [-1.0] * 7 + [0.0] * 3
        result = TTSAnalyzer().from_samples(energies, ground_state_energy=-1.0)
        assert result["p_success"] == 0.7
        assert result["tts_us"] < 200.0  # should be fast with 70% success

    def test_compare_solvers(self) -> None:
        results = {
            "sa": {"energies": [-1.0] * 8 + [0.0] * 2, "t_anneal_us": 100.0},
            "qpu": {"energies": [-1.0] * 5 + [0.0] * 5, "t_anneal_us": 20.0},
        }
        comparison = TTSAnalyzer().compare_solvers(results, ground_state_energy=-1.0)
        assert "sa" in comparison
        assert "qpu" in comparison
