# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (backend_parity) from former test_expif_backends.py

from __future__ import annotations

from tests.expif_backends_support import *  # noqa: F403


def test_every_acceleration_backend_is_executable() -> None:
    """A fidelity-closure run exposes all four real compiled lanes without skips."""
    assert expif._HAS_RUST
    assert expif._ensure_julia_loaded()
    assert expif._ensure_go_loaded()
    assert expif._ensure_mojo_loaded()


def test_missing_rust_engine_is_detected_at_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the optional-engine import boundary without leaving module drift."""
    real_import = importlib.import_module

    def without_engine(name: str, package: str | None = None) -> object:
        if name == "sc_neurocore_engine":
            raise ImportError("engine intentionally hidden")
        return real_import(name, package)

    with monkeypatch.context() as patch:
        patch.setattr(importlib, "import_module", without_engine)
        reloaded = importlib.reload(expif)
        assert reloaded._HAS_RUST is False
        assert reloaded._EngineExpIFCls is None
    importlib.reload(expif)
    assert expif._HAS_RUST is True


@pytest.mark.parametrize(("current", "expected_spikes"), _GOLDENS)
@pytest.mark.parametrize("backend", _COMPILED_BACKENDS)
def test_compiled_backends_match_python_golden(
    backend: str,
    current: float,
    expected_spikes: int,
) -> None:
    """Preserve the complete trace, final state, and source-bound events."""
    reference_trace, reference_spikes, reference_state = _run("python", current=current)
    trace, spikes, state = _run(backend, current=current)

    assert reference_spikes == expected_spikes
    assert spikes == reference_spikes
    np.testing.assert_allclose(trace, reference_trace, atol=_TRACE_ATOL, rtol=0.0)
    np.testing.assert_allclose(state, reference_state, atol=_TRACE_ATOL, rtol=0.0)


@pytest.mark.parametrize("backend", ("julia", "go", "mojo"))
def test_full_parameter_and_refractory_contract_matches_python(backend: str) -> None:
    """Carry every maintained numeric field across full-parameter native ABIs."""
    reference_trace, reference_spikes, reference_state = _run(
        "python", current=50.0, n_steps=500, factory=_configured
    )
    trace, spikes, state = _run(backend, current=50.0, n_steps=500, factory=_configured)
    assert spikes == reference_spikes == 2
    np.testing.assert_allclose(trace, reference_trace, atol=_TRACE_ATOL, rtol=0.0)
    np.testing.assert_allclose(state, reference_state, atol=_TRACE_ATOL, rtol=0.0)


@pytest.mark.parametrize("backend", _COMPILED_BACKENDS)
def test_empty_run_preserves_state(backend: str) -> None:
    """Return an empty trace without discarding voltage or refractory state."""
    neuron = ExpIFNeuron() if backend == "rust" else _configured()
    before = (neuron.v, neuron.refractory_remaining)
    trace, spikes = neuron.simulate(0, 20.0, backend=backend)
    assert trace.shape == (0,)
    assert spikes == 0
    assert (neuron.v, neuron.refractory_remaining) == before
