# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quantum-annealing solver and backend tests

"""Exercise optional dependency loading and solver result contracts."""

from __future__ import annotations

import builtins
import importlib
import sys
import types
from collections.abc import Mapping, Sequence

import pytest

from sc_neurocore.bridges import annealing_backends as backends
from sc_neurocore.bridges.quantum_annealing import (
    DWaveInterface,
    IsingModel,
    QUBOModel,
    SimulatedAnnealer,
)
from tests.test_bridges.quantum_annealing_test_helpers import simple_ising, unsafe


def test_optional_import_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing dimod and D-Wave imports leave explicit unavailable state."""
    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> object:
        if name == "dimod" or name == "dwave.system" or name.startswith("dwave."):
            raise ImportError(name)
        return real_import(name, globals, locals, fromlist, level)

    try:
        monkeypatch.setattr(builtins, "__import__", guarded_import)
        module = importlib.reload(backends)
        assert module.HAS_DIMOD is False
        assert module.dimod is None
        assert module.HAS_DWAVE is False
        assert module.DWaveSampler is None
        assert module.EmbeddingComposite is None
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)
        importlib.reload(backends)


def test_optional_import_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importable optional modules populate the backend constructors."""
    fake_dimod = types.ModuleType("dimod")
    fake_dwave = types.ModuleType("dwave")
    fake_system = types.ModuleType("dwave.system")

    class FakeSampler:
        pass

    class FakeComposite:
        pass

    fake_system.__dict__["DWaveSampler"] = FakeSampler
    fake_system.__dict__["EmbeddingComposite"] = FakeComposite
    fake_dwave.__dict__["system"] = fake_system
    try:
        monkeypatch.setitem(sys.modules, "dimod", fake_dimod)
        monkeypatch.setitem(sys.modules, "dwave", fake_dwave)
        monkeypatch.setitem(sys.modules, "dwave.system", fake_system)
        module = importlib.reload(backends)
        assert module.HAS_DIMOD is True
        assert module.dimod is fake_dimod
        assert module.HAS_DWAVE is True
        assert module.DWaveSampler is FakeSampler
        assert module.EmbeddingComposite is FakeComposite
    finally:
        sys.modules.pop("dimod", None)
        sys.modules.pop("dwave", None)
        sys.modules.pop("dwave.system", None)
        importlib.reload(backends)


def test_native_import_fallback_clears_incomplete_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing native module leaves every native capability disabled."""
    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> object:
        if name == "sc_neurocore_engine.quantum":
            raise ImportError(name)
        return real_import(name, globals, locals, fromlist, level)

    try:
        monkeypatch.setattr(builtins, "__import__", guarded_import)
        module = importlib.reload(backends)
        assert module.HAS_RUST_QA is False
        assert module._rust_ising_energy is None
        assert module._rust_batch_energy is None
        assert module._rust_simulated_annealing is None
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)
        importlib.reload(backends)


def test_backend_require_helpers_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit missing optional backends raise stable errors."""
    monkeypatch.setattr(backends, "HAS_RUST_QA", False)
    monkeypatch.setattr(backends, "HAS_DIMOD", False)
    monkeypatch.setattr(backends, "HAS_DWAVE", False)
    assert backends.build_spin_bqm({}, {}, 0.0) is None
    with pytest.raises(RuntimeError, match="energy backend"):
        backends.require_rust_energy()
    with pytest.raises(RuntimeError, match="batch backend"):
        backends.require_rust_batch_energy()
    with pytest.raises(RuntimeError, match="solver backend"):
        backends.require_rust_annealer()
    with pytest.raises(RuntimeError, match="Ocean SDK"):
        backends.require_dwave_components()


def test_backend_require_helpers_return_configured_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured native kernels and dimod objects are returned unchanged."""
    energy = unsafe(lambda *args: 0.0)
    batch = unsafe(lambda *args: [0.0])
    annealer = unsafe(lambda *args: {})

    class FakeDimod:
        @staticmethod
        def BinaryQuadraticModel(*args: object) -> tuple[object, ...]:
            return args

    class FakeSampler:
        pass

    class FakeComposite:
        pass

    monkeypatch.setattr(backends, "HAS_RUST_QA", True)
    monkeypatch.setattr(backends, "_rust_ising_energy", energy)
    monkeypatch.setattr(backends, "_rust_batch_energy", batch)
    monkeypatch.setattr(backends, "_rust_simulated_annealing", annealer)
    monkeypatch.setattr(backends, "HAS_DIMOD", True)
    monkeypatch.setattr(backends, "HAS_DWAVE", True)
    monkeypatch.setattr(backends, "dimod", FakeDimod)
    monkeypatch.setattr(backends, "DWaveSampler", FakeSampler)
    monkeypatch.setattr(backends, "EmbeddingComposite", FakeComposite)

    assert backends.require_rust_energy() is energy
    assert backends.require_rust_batch_energy() is batch
    assert backends.require_rust_annealer() is annealer
    assert backends.build_spin_bqm({0: 1.0}, {}, 0.5) == (
        {0: 1.0},
        {},
        0.5,
        "SPIN",
    )
    assert backends.require_dwave_components() == (FakeDimod, FakeSampler, FakeComposite)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_sweeps": 0}, "n_sweeps"),
        ({"n_sweeps": unsafe(True)}, "n_sweeps"),
        ({"beta_start": 0.0}, "beta_start"),
        ({"beta_end": float("nan")}, "beta_end"),
        ({"beta_start": 2.0, "beta_end": 1.0}, "beta_end"),
        ({"seed": unsafe(1.5)}, "seed"),
        ({"backend": unsafe("gpu")}, "backend"),
    ],
)
def test_simulated_annealer_rejects_invalid_configuration(
    kwargs: dict[str, object], match: str
) -> None:
    """Annealing schedule and backend configuration fail closed."""
    with pytest.raises(ValueError, match=match):
        SimulatedAnnealer(**unsafe(kwargs))


def test_python_solver_is_deterministic_and_finds_ground_state() -> None:
    """Seeded Python runs preserve the sample contract and find a simple ground state."""
    model = IsingModel(h={0: 0.0, 1: 0.0}, J={(0, 1): -1.0})
    first = SimulatedAnnealer(n_sweeps=200, seed=7, backend="python").solve_ising(
        model, num_reads=5
    )
    second = SimulatedAnnealer(n_sweeps=200, seed=7, backend="python").solve_ising(
        model, num_reads=5
    )
    assert first == second
    assert first["backend"] == "python"
    assert first["best_energy"] == pytest.approx(-1.0)
    assert len(first["samples"]) == 5
    assert all(spin in {-1, 1} for spin in first["best_spins"].values())


def test_one_sweep_python_solver_and_qubo_mapping() -> None:
    """The single-sweep branch and QUBO bit conversion remain valid."""
    qubo = QUBOModel(Q={(0, 0): -1.0, (1, 1): -1.0, (0, 1): 2.0})
    result = SimulatedAnnealer(n_sweeps=1, seed=42, backend="python").solve_qubo(qubo, num_reads=3)
    assert result["backend"] == "python"
    assert len(result["samples"]) == 3
    assert result["best_energy"] == qubo.energy(result["best_bits"])
    assert all(bit in {0, 1} for bit in result["best_bits"].values())


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: SimulatedAnnealer().solve_ising(unsafe("bad")), "IsingModel"),
        (lambda: SimulatedAnnealer().solve_ising(IsingModel()), "one qubit"),
        (lambda: SimulatedAnnealer().solve_ising(simple_ising(), 0), "num_reads"),
        (lambda: SimulatedAnnealer().solve_qubo(unsafe("bad")), "QUBOModel"),
    ],
)
def test_solver_rejects_invalid_calls(call: object, match: str) -> None:
    """Solver entry points validate models and read counts."""
    with pytest.raises(ValueError, match=match):
        unsafe(call)()


def _valid_native_result(size: int) -> dict[str, object]:
    """Return a complete two-sample native result."""
    first = [1 if index % 2 == 0 else -1 for index in range(size)]
    second = [-spin for spin in first]
    return {
        "best_spins": first,
        "best_energy": -3.0,
        "energies": [-3.0, -2.0],
        "samples": [first, second],
    }


def test_native_solver_contract_and_seed_forwarding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Native dispatch validates and maps arrays while forwarding the configured seed."""
    captured: tuple[object, ...] = ()

    def fake_solver(*args: object) -> Mapping[str, object]:
        nonlocal captured
        captured = args
        return _valid_native_result(12)

    monkeypatch.setattr(backends, "HAS_RUST_QA", True)
    monkeypatch.setattr(backends, "_rust_simulated_annealing", fake_solver)
    model = IsingModel(
        h={0: 0.5, 2: -0.25},
        J={(0, 1): -1.0, (1, 2): 0.75},
        offset=0.5,
        n_qubits=12,
    )
    result = SimulatedAnnealer(
        n_sweeps=17,
        beta_start=0.2,
        beta_end=3.0,
        seed=91,
        backend="rust",
    ).solve_ising(model, 2)
    assert result["backend"] == "rust"
    assert result["best_spins"][1] == -1
    assert result["samples"][1][0] == -1
    assert captured[-1] == 91
    assert captured[:7] == (
        [0, 2],
        [0.5, -0.25],
        [0, 1],
        [1, 2],
        [-1.0, 0.75],
        12,
        0.5,
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda result: result.update(best_spins="bad"), "best_spins"),
        (lambda result: result.update(best_spins=[1]), "best_spins"),
        (lambda result: result.update(samples="bad"), "samples"),
        (lambda result: result.update(samples=[[0] * 12]), "samples"),
        (lambda result: result.update(energies="bad"), "energies"),
        (lambda result: result.update(energies=[float("nan"), -2.0]), "non-finite"),
        (lambda result: result.update(samples=[[1] * 12], energies=[1.0, 2.0]), "mismatched"),
        (lambda result: result.update(best_energy=True), "best_energy"),
    ],
)
def test_native_solver_rejects_malformed_results(
    monkeypatch: pytest.MonkeyPatch,
    mutation: object,
    match: str,
) -> None:
    """Malformed native payloads cannot cross the Python boundary."""
    result = _valid_native_result(12)
    unsafe(mutation)(result)
    monkeypatch.setattr(backends, "HAS_RUST_QA", True)
    monkeypatch.setattr(backends, "_rust_simulated_annealing", lambda *args: result)
    with pytest.raises(RuntimeError, match=match):
        SimulatedAnnealer(backend="rust").solve_ising(IsingModel(h={0: 0.0}, n_qubits=12), 2)


def test_explicit_missing_native_solver_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit native solver request never changes backend silently."""
    monkeypatch.setattr(backends, "HAS_RUST_QA", False)
    with pytest.raises(RuntimeError, match="unavailable"):
        SimulatedAnnealer(backend="rust").solve_ising(simple_ising())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"chain_strength": 0.0}, "chain_strength"),
        ({"num_reads": unsafe(False)}, "num_reads"),
        ({"annealing_time_us": float("inf")}, "annealing_time_us"),
    ],
)
def test_dwave_interface_rejects_invalid_configuration(
    kwargs: dict[str, object], match: str
) -> None:
    """QPU parameters must be finite and positive."""
    with pytest.raises(ValueError, match=match):
        DWaveInterface(**unsafe(kwargs))


def test_dwave_fallback_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing Ocean dependencies use no more than 20 local reads."""
    monkeypatch.setattr(backends, "HAS_DWAVE", False)
    monkeypatch.setattr(backends, "HAS_DIMOD", False)
    interface = DWaveInterface(num_reads=50)
    assert interface.available is False
    result = interface.solve_ising(simple_ising())
    assert result["backend"] == "simulated_annealing_fallback"
    assert result["num_reads"] == 20
    with pytest.raises(ValueError, match="non-empty"):
        interface.solve_ising(unsafe("bad"))


def test_dwave_qpu_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """QPU submission forwards the BQM and reports validated timing."""
    captured: dict[str, object] = {}

    class FakeDimod:
        class BinaryQuadraticModel:
            def __init__(self, *args: object) -> None:
                captured["bqm_args"] = args

    class FakeSampler:
        pass

    class FakeBest:
        sample = {0: 1, 1: -1, 2: 1}
        energy = -1.25

    class FakeResponse:
        first = FakeBest()
        info = {"timing": {"qpu_access_time": 123}}

    class FakeComposite:
        def __init__(self, sampler: FakeSampler) -> None:
            captured["sampler"] = sampler

        def sample(self, bqm: object, **kwargs: object) -> FakeResponse:
            captured["bqm"] = bqm
            captured["kwargs"] = kwargs
            return FakeResponse()

    monkeypatch.setattr(backends, "HAS_DWAVE", True)
    monkeypatch.setattr(backends, "HAS_DIMOD", True)
    monkeypatch.setattr(backends, "dimod", FakeDimod)
    monkeypatch.setattr(backends, "DWaveSampler", FakeSampler)
    monkeypatch.setattr(backends, "EmbeddingComposite", FakeComposite)
    model = simple_ising()
    result = DWaveInterface(1.7, 31, 23.0).solve_ising(model)
    assert result == {
        "best_spins": {0: 1, 1: -1, 2: 1},
        "best_energy": -1.25,
        "num_reads": 31,
        "backend": "dwave_qpu",
        "timing": {"qpu_access_time": 123},
    }
    assert captured["bqm_args"] == (model.h, model.J, model.offset, "SPIN")
    assert captured["kwargs"] == {
        "num_reads": 31,
        "chain_strength": 1.7,
        "annealing_time": 23.0,
    }


@pytest.mark.parametrize(
    ("first", "info", "match"),
    [
        (unsafe(None), {}, "best sample"),
        (unsafe(types.SimpleNamespace(sample="bad", energy=-1.0)), {}, "best sample"),
        (unsafe(types.SimpleNamespace(sample={0: 1}, energy=float("inf"))), {}, "non-finite"),
    ],
)
def test_dwave_rejects_malformed_response(
    monkeypatch: pytest.MonkeyPatch,
    first: object,
    info: object,
    match: str,
) -> None:
    """Incomplete QPU responses raise instead of fabricating results."""

    class FakeDimod:
        class BinaryQuadraticModel:
            def __init__(self, *args: object) -> None:
                self.args = args

    class FakeSampler:
        pass

    class FakeResponse:
        def __init__(self) -> None:
            self.first = first
            self.info = info

    class FakeComposite:
        def __init__(self, sampler: object) -> None:
            self.sampler = sampler

        def sample(self, bqm: object, **kwargs: object) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(backends, "HAS_DWAVE", True)
    monkeypatch.setattr(backends, "HAS_DIMOD", True)
    monkeypatch.setattr(backends, "dimod", FakeDimod)
    monkeypatch.setattr(backends, "DWaveSampler", FakeSampler)
    monkeypatch.setattr(backends, "EmbeddingComposite", FakeComposite)
    with pytest.raises(RuntimeError, match=match):
        DWaveInterface().solve_ising(IsingModel(h={0: 0.0}))
