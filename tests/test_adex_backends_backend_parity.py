# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (backend_parity) from former test_adex_backends.py

from __future__ import annotations

from tests.adex_backends_support import *  # noqa: F403


def test_every_acceleration_backend_is_executable() -> None:
    """A graduation run must expose all four real compiled lanes without skips."""
    assert adex._HAS_RUST
    assert adex._ensure_julia_loaded()
    assert adex._ensure_go_loaded()
    assert adex._ensure_mojo_loaded()


def test_missing_rust_engine_is_detected_at_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the optional-engine import boundary without leaving module drift."""
    original_namespace = adex.__dict__.copy()
    try:
        real_import = importlib.import_module

        def without_engine(name: str, package: str | None = None) -> object:
            if name == "sc_neurocore_engine":
                raise ImportError("engine intentionally hidden")
            return real_import(name, package)

        with monkeypatch.context() as patch:
            patch.setattr(importlib, "import_module", without_engine)
            reloaded = importlib.reload(adex)
            assert reloaded._HAS_RUST is False
            assert reloaded._EngineAdExCls is None
            assert reloaded._EngineAdExSimulateFn is None
        importlib.reload(adex)
        assert adex._HAS_RUST is True
    finally:
        # Reload mutates the shared module in place and rebinds AdExNeuron.
        # Restore its original namespace so later tests retain their imported
        # class identity instead of observing a stale pre-reload class.
        adex.__dict__.clear()
        adex.__dict__.update(original_namespace)

    assert adex.AdExNeuron is AdExNeuron


def test_invalid_integrator_and_runtime_voltage_fail_closed() -> None:
    """Cover constructor and dynamic-voltage validation boundaries."""
    invalid = cast(Literal["baseline_euler", "rk4", "rosenbrock"], "invalid")
    with pytest.raises(ValueError, match="Unsupported integrator"):
        AdExNeuron(integrator=invalid)

    neuron = AdExNeuron()
    neuron.v = math.nan
    with pytest.raises(ValueError, match="runtime voltage"):
        neuron.simulate(1, 0.0, backend="python")


@pytest.mark.parametrize("current,expected_spikes", _GOLDENS)
@pytest.mark.parametrize("backend", _COMPILED_BACKENDS)
def test_compiled_backends_match_python_golden(
    backend: str,
    current: float,
    expected_spikes: int,
) -> None:
    """Preserve the complete trace, final state and stable event observable."""
    reference_trace, reference_spikes, reference_state = _run("python", current=current)
    trace, spikes, state = _run(backend, current=current)

    assert reference_spikes == expected_spikes
    assert spikes == reference_spikes
    np.testing.assert_allclose(trace, reference_trace, atol=_TRACE_ATOL, rtol=0.0)
    np.testing.assert_allclose(state, reference_state, atol=_TRACE_ATOL, rtol=0.0)


@pytest.mark.parametrize("backend", _COMPILED_BACKENDS)
def test_full_parameter_contract_matches_python(backend: str) -> None:
    """Carry every maintained numeric field across non-default native ABIs."""

    def configured() -> AdExNeuron:
        return AdExNeuron(
            v=-60.0,
            w=3.0,
            v_rest=-64.0,
            v_reset=-69.0,
            v_threshold=-49.0,
            v_rh=-54.0,
            delta_t=2.5,
            tau=18.0,
            tau_w=120.0,
            a=0.7,
            b=8.0,
            c_m=180.0,
            dt=0.2,
        )

    reference_trace, reference_spikes, reference_state = _run(
        "python", current=410.0, n_steps=250, factory=configured
    )
    trace, spikes, state = _run(backend, current=410.0, n_steps=250, factory=configured)
    assert spikes == reference_spikes == 5
    np.testing.assert_allclose(trace, reference_trace, atol=_TRACE_ATOL, rtol=0.0)
    np.testing.assert_allclose(state, reference_state, atol=_TRACE_ATOL, rtol=0.0)


@pytest.mark.parametrize("backend", _COMPILED_BACKENDS)
def test_empty_run_preserves_state(backend: str) -> None:
    """Return an empty trace without discarding either state variable."""
    neuron = AdExNeuron(v=-60.0, w=3.0)
    before = (neuron.v, neuron.w)
    trace, spikes = neuron.simulate(0, 250.0, backend=backend)
    assert trace.shape == (0,)
    assert spikes == 0
    assert (neuron.v, neuron.w) == before
