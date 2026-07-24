# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (optional_imports) from former test_quantum_annealing_solvers_backends.py

from __future__ import annotations

from quantum_annealing_solvers_backends_support import *  # noqa: F403


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
