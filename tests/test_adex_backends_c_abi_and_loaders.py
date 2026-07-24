# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (c_abi_and_loaders) from former test_adex_backends.py

from __future__ import annotations

from tests.adex_backends_support import *  # noqa: F403

@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_c_abi_rejects_non_finite_input_without_writing_output(backend: str) -> None:
    """Prove invalid input is rejected inside each C boundary, not only in Python."""
    neuron = AdExNeuron()
    output = np.full(3, -999.0, dtype=np.float64)
    if backend == "go":
        assert adex._go_lib is not None
        result = adex._go_lib.adex_simulate_c(
            *_c_arguments(neuron),
            1,
            math.nan,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
    else:
        assert adex._mojo_lib is not None
        result = adex._mojo_lib.adex_simulate_c(
            *_c_arguments(neuron), 1, math.nan, int(output.ctypes.data)
        )
    assert result == -1
    np.testing.assert_array_equal(output, np.full(3, -999.0, dtype=np.float64))


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_c_abi_rejection_does_not_commit_instance_state(backend: str) -> None:
    """Translate a native non-finite candidate into a mutation-free failure."""
    neuron = AdExNeuron(dt=1.0e308)
    before = (neuron.v, neuron.w)
    with pytest.raises(FloatingPointError, match="kernel rejected"):
        neuron.simulate(1, 1.0e308, backend=backend)
    assert (neuron.v, neuron.w) == before


@pytest.mark.parametrize("backend", ("julia", "go", "mojo"))
def test_requested_backend_reports_unavailable(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return an actionable failure instead of silently falling back to Python."""
    monkeypatch.setattr(adex, f"_ensure_{backend}_loaded", lambda: False)
    with pytest.raises(RuntimeError, match=backend.title()):
        AdExNeuron().simulate(1, 0.0, backend=backend)


def test_requested_rust_backend_reports_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep explicit Rust requests fail-closed when the engine wheel is absent."""
    monkeypatch.setattr(adex, "_HAS_RUST", False)
    monkeypatch.setattr(adex, "_EngineAdExCls", None)
    with pytest.raises(RuntimeError, match="Rust AdEx backend"):
        AdExNeuron().simulate(1, 0.0, backend="rust")


@pytest.mark.parametrize("backend", ("go", "mojo"))
@pytest.mark.parametrize("failure", ("missing", "load", "symbol"))
def test_c_backend_loader_rejects_invalid_library_boundaries(
    backend: str,
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep absent, unloadable or symbol-incomplete libraries unavailable."""
    monkeypatch.setattr(adex, f"_{backend}_lib", None)
    monkeypatch.setattr(adex, f"_HAS_{backend.upper()}", False)
    monkeypatch.setattr(os.path, "isfile", lambda _path: failure != "missing")
    if failure == "load":

        def reject_load(_path: str) -> object:
            raise OSError("invalid shared library")

        monkeypatch.setattr(ctypes, "CDLL", reject_load)
    elif failure == "symbol":
        monkeypatch.setattr(ctypes, "CDLL", lambda _path: object())

    assert getattr(adex, f"_ensure_{backend}_loaded")() is False
    assert getattr(adex, f"_{backend}_lib") is None
    assert getattr(adex, f"_HAS_{backend.upper()}") is False


@pytest.mark.parametrize("failure", ("missing", "load", "module"))
def test_julia_loader_rejects_invalid_runtime_boundaries(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep missing runtimes, source files and broken modules unavailable."""
    monkeypatch.setattr(adex, "_julia_module", None)
    monkeypatch.setattr(adex, "_HAS_JULIA", False)
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda _name: None if failure == "missing" else 1
    )
    monkeypatch.setattr(os.path, "isfile", lambda _path: failure != "load")
    if failure == "module":

        def reject_import(_name: str) -> object:
            raise RuntimeError("broken Julia runtime")

        monkeypatch.setattr(importlib, "import_module", reject_import)

    assert adex._ensure_julia_loaded() is False
    assert adex._julia_module is None
    assert adex._HAS_JULIA is False
