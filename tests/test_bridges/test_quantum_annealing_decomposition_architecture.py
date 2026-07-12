# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quantum-annealing decomposition and architecture tests

"""Pin overlap semantics, facade compatibility, DAG shape, and source truth."""

from __future__ import annotations

import ast
import pickle
from pathlib import Path
from typing import Any

import pytest

import sc_neurocore.bridges as bridges
import sc_neurocore.bridges.annealing_analysis as analysis
import sc_neurocore.bridges.annealing_compilers as compilers
import sc_neurocore.bridges.annealing_decomposition as decomposition
import sc_neurocore.bridges.annealing_hardware as hardware
import sc_neurocore.bridges.annealing_io as annealing_io
import sc_neurocore.bridges.annealing_models as models
import sc_neurocore.bridges.annealing_solvers as solvers
import sc_neurocore.bridges.annealing_transforms as transforms
import sc_neurocore.bridges.quantum_annealing as facade
from sc_neurocore.bridges.quantum_annealing import (
    IsingModel,
    ProblemDecomposer,
    SimulatedAnnealer,
)
from tests.test_bridges.quantum_annealing_test_helpers import simple_ising, unsafe


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BRIDGES_ROOT = _REPO_ROOT / "src" / "sc_neurocore" / "bridges"
_RESPONSIBILITY_MODULES = (
    "annealing_backends",
    "annealing_models",
    "annealing_compilers",
    "annealing_solvers",
    "annealing_analysis",
    "annealing_hardware",
    "annealing_transforms",
    "annealing_io",
    "annealing_decomposition",
)


class FixedSpinSolver(SimulatedAnnealer):
    """Return a deterministic local spin pattern for reconstruction tests."""

    def __init__(self, spin: int = -1) -> None:
        super().__init__(n_sweeps=1, backend="python")
        self._fixed_spin = spin

    def solve_ising(self, model: IsingModel, num_reads: int = 10) -> dict[str, Any]:
        """Return the configured spin for every local qubit."""
        sample = {index: self._fixed_spin for index in range(model.n_qubits)}
        return {
            "best_spins": sample,
            "best_energy": model.energy(sample, backend="python"),
            "samples": [sample],
            "energies": [model.energy(sample, backend="python")],
        }


class MalformedSolver(SimulatedAnnealer):
    """Return a supplied invalid mapping to exercise the trust boundary."""

    def __init__(self, best_spins: object) -> None:
        super().__init__(n_sweeps=1, backend="python")
        self._best_spins = best_spins

    def solve_ising(self, model: IsingModel, num_reads: int = 10) -> dict[str, Any]:
        """Return an intentionally malformed result."""
        return {"best_spins": self._best_spins}


def _chain_model(size: int) -> IsingModel:
    """Return an unlabeled ferromagnetic chain."""
    return IsingModel(
        h={index: 0.0 for index in range(size)},
        J={(index, index + 1): -1.0 for index in range(size - 1)},
        n_qubits=size,
        source="chain",
    )


def test_small_decomposition_retains_identity() -> None:
    """A model that already fits is returned without a lossy copy."""
    model = simple_ising()
    assert ProblemDecomposer(max_subproblem_size=4, overlap=1).decompose(model) == [model]
    assert ProblemDecomposer(max_subproblem_size=4, overlap=1).decompose(model)[0] is model


def test_forced_decomposition_honors_size_overlap_and_coverage() -> None:
    """Connected partitions share real boundary qubits and cover every global qubit."""
    model = _chain_model(10)
    parts = ProblemDecomposer(max_subproblem_size=4, overlap=1).decompose(model)
    assert len(parts) > 1
    assert all(part.n_qubits <= 4 for part in parts)
    label_sets = [set(part.qubit_labels.values()) for part in parts]
    assert set().union(*label_sets) == {f"q{index}" for index in range(10)}
    assert any(first.intersection(second) for first, second in zip(label_sets, label_sets[1:]))


def test_disconnected_decomposition_is_deterministic() -> None:
    """Disconnected qubits use stable ascending fallback partitioning."""
    model = IsingModel(h={index: float(index) for index in range(7)}, n_qubits=7)
    parts = ProblemDecomposer(max_subproblem_size=3, overlap=0).decompose(model)
    assert [list(part.h.values()) for part in parts] == [
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [6.0],
    ]


def test_solve_decomposed_maps_unlabeled_qubits_by_index() -> None:
    """Reconstruction does not depend on optional or duplicated display labels."""
    model = _chain_model(8)
    result = ProblemDecomposer(
        max_subproblem_size=3,
        overlap=1,
        n_iterations=2,
    ).solve_decomposed(model, FixedSpinSolver(-1))
    assert result["best_spins"] == {index: -1 for index in range(8)}
    assert result["best_energy"] == model.energy(result["best_spins"], backend="python")
    assert result["n_partitions"] > 1
    assert result["n_iterations"] == 2


def test_solve_decomposed_small_model_path() -> None:
    """The one-partition path maps local indices without rebuilding the model."""
    result = ProblemDecomposer(max_subproblem_size=4, overlap=1).solve_decomposed(
        simple_ising(), FixedSpinSolver(1)
    )
    assert result["best_spins"] == {0: 1, 1: 1, 2: 1}
    assert result["n_partitions"] == 1


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: ProblemDecomposer(0), "max_subproblem_size"),
        (lambda: ProblemDecomposer(overlap=unsafe(True)), "overlap"),
        (lambda: ProblemDecomposer(max_subproblem_size=4, overlap=4), "smaller"),
        (lambda: ProblemDecomposer(n_iterations=0), "n_iterations"),
        (lambda: ProblemDecomposer().decompose(unsafe("bad")), "non-empty"),
        (lambda: ProblemDecomposer().solve_decomposed(IsingModel()), "non-empty"),
        (lambda: ProblemDecomposer().solve_decomposed(simple_ising(), unsafe("bad")), "solver"),
    ],
)
def test_decomposer_rejects_invalid_inputs(call: object, match: str) -> None:
    """Partition configuration, models, and solver types fail closed."""
    with pytest.raises(ValueError, match=match):
        unsafe(call)()


@pytest.mark.parametrize(
    ("best_spins", "match"),
    [
        (unsafe("bad"), "no best_spins"),
        ({unsafe(True): 1}, "invalid spin mapping"),
        ({99: 1}, "invalid spin mapping"),
        ({0: 0}, "invalid spin mapping"),
    ],
)
def test_decomposer_rejects_untrusted_solver_results(
    best_spins: object,
    match: str,
) -> None:
    """Subproblem results are validated before global reconstruction."""
    with pytest.raises(RuntimeError, match=match):
        ProblemDecomposer(max_subproblem_size=2, overlap=0).solve_decomposed(
            _chain_model(3),
            MalformedSolver(best_spins),
        )


def test_public_facade_and_bridges_exports_are_stable() -> None:
    """All 24 historical symbols retain facade and package identity."""
    assert len(facade.__all__) == 24
    assert len(set(facade.__all__)) == 24
    for name in facade.__all__:
        symbol = getattr(facade, name)
        assert getattr(bridges, name) is symbol
        assert symbol.__module__ == facade.__name__
        assert pickle.loads(pickle.dumps(symbol)) is symbol


def test_responsibility_owners_are_exact() -> None:
    """Every public implementation has one expected responsibility owner."""
    expected = {
        models: {"ProblemType", "QubitSpec", "CouplerSpec", "IsingModel", "QUBOModel"},
        compilers: {"SCToIsing", "SCToQUBO", "SCBitstreamQUBO"},
        solvers: {"SimulatedAnnealer", "DWaveInterface"},
        analysis: {"EnergyLandscape", "EmbeddingAnalyzer", "SampleAggregator", "TTSAnalyzer"},
        hardware: {"HardwareGraph", "ChainBreakResolver"},
        transforms: {"AnnealingSchedule", "GaugeTransform", "SCPrecisionEncoder"},
        annealing_io: {"export_ising_json", "export_qubo_json", "export_bqm", "visualize_ising"},
        decomposition: {"ProblemDecomposer"},
    }
    observed: set[str] = set()
    for owner, names in expected.items():
        for name in names:
            assert getattr(owner, name) is getattr(facade, name)
        observed.update(names)
    assert observed == set(facade.__all__)


def _responsibility_import_graph() -> dict[str, set[str]]:
    """Parse responsibility-module imports into a local graph."""
    graph: dict[str, set[str]] = {name: set() for name in _RESPONSIBILITY_MODULES}
    prefix = "sc_neurocore.bridges."
    for name in _RESPONSIBILITY_MODULES:
        tree = ast.parse((_BRIDGES_ROOT / f"{name}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported = node.module.removeprefix(prefix)
                if imported in graph:
                    graph[name].add(imported)
                assert imported != "quantum_annealing"
    return graph


def test_responsibility_graph_is_acyclic() -> None:
    """Extracted modules form a DAG and never import the compatibility facade."""
    graph = _responsibility_import_graph()
    temporary: set[str] = set()
    permanent: set[str] = set()

    def visit(name: str) -> None:
        if name in permanent:
            return
        assert name not in temporary, f"annealing responsibility cycle at {name}"
        temporary.add(name)
        for dependency in graph[name]:
            visit(dependency)
        temporary.remove(name)
        permanent.add(name)

    for module_name in graph:
        visit(module_name)


def test_source_and_test_size_caps() -> None:
    """The facade, responsibility modules, and focused tests remain bounded."""
    facade_lines = (_BRIDGES_ROOT / "quantum_annealing.py").read_text(encoding="utf-8").count("\n")
    assert facade_lines <= 120
    for name in _RESPONSIBILITY_MODULES:
        line_count = (_BRIDGES_ROOT / f"{name}.py").read_text(encoding="utf-8").count("\n")
        assert line_count <= 400, (name, line_count)
    for path in Path(__file__).parent.glob("test_quantum_annealing*.py"):
        assert path.read_text(encoding="utf-8").count("\n") <= 500, path


def test_false_polyglot_mirrors_are_absent_and_native_engine_remains() -> None:
    """Generated report mirrors are gone; the maintained native kernel stays authoritative."""
    removed = (
        "src/sc_neurocore/accel/rust/safety/quantum_annealing.rs",
        "src/sc_neurocore/accel/go/services/quantum_annealing/quantum_annealing.go",
        "src/sc_neurocore/accel/go/services/quantum_annealing/__init__.py",
        "src/sc_neurocore/accel/julia/bridges/quantum_annealing.jl",
        "src/sc_neurocore/accel/mojo/kernels/quantum_annealing.mojo",
    )
    assert not [path for path in removed if (_REPO_ROOT / path).exists()]
    engine = (_REPO_ROOT / "engine" / "src" / "quantum.rs").read_text(encoding="utf-8")
    assert "pub fn ising_energy(" in engine
    assert "pub fn batch_ising_energy(" in engine
    assert "pub fn simulated_annealing(" in engine
    assert engine.count("#[test]") >= 10
