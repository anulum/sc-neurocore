# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (auto_dispatch) from former test_dpi_neuron_backends.py

from __future__ import annotations

from tests.dpi_neuron_backends_support import *  # noqa: F403


def test_rust_rejects_non_default_contract_without_mutation() -> None:
    """Fail closed outside the engine's fixed-constructor compatibility boundary."""
    neuron = _configured()
    before = (neuron.i_mem, neuron.i_ahp, neuron.refractory_time)
    with pytest.raises(RuntimeError, match="factory-default"):
        neuron.simulate(1, 0.0, backend="rust")
    assert (neuron.i_mem, neuron.i_ahp, neuron.refractory_time) == before


def test_auto_prefers_go_without_initialising_other_runtimes() -> None:
    """Route through Go without initialising Julia or probing Mojo."""
    with (
        patch.object(backends, "ensure_julia_loaded") as ensure_julia,
        patch.object(backends, "ensure_mojo_loaded") as ensure_mojo,
    ):
        auto = _configured()
        expected = _configured()
        actual_trace, actual_spikes = auto.simulate(100, 5.0, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 5.0, backend="go")
    ensure_julia.assert_not_called()
    ensure_mojo.assert_not_called()
    np.testing.assert_array_equal(actual_trace, expected_trace)
    assert (actual_spikes, auto.i_mem, auto.i_ahp) == (
        expected_spikes,
        expected.i_mem,
        expected.i_ahp,
    )


def test_auto_falls_through_julia_mojo_rust_and_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the complete documented fallback chain."""
    monkeypatch.setattr(backends, "ensure_go_loaded", lambda: False)
    julia_auto, julia_expected = _configured(), _configured()
    actual, events = julia_auto.simulate(100, 5.0)
    expected, expected_events = julia_expected.simulate(100, 5.0, backend="julia")
    np.testing.assert_array_equal(actual, expected)
    assert events == expected_events

    monkeypatch.setattr(backends, "ensure_julia_loaded", lambda: False)
    mojo_auto, mojo_expected = _configured(), _configured()
    actual, events = mojo_auto.simulate(100, 5.0)
    expected, expected_events = mojo_expected.simulate(100, 5.0, backend="mojo")
    np.testing.assert_array_equal(actual, expected)
    assert events == expected_events

    monkeypatch.setattr(backends, "ensure_mojo_loaded", lambda: False)
    rust_auto, rust_expected = DPINeuron(), DPINeuron()
    actual, events = rust_auto.simulate(100, 5.0)
    expected, expected_events = rust_expected.simulate(100, 5.0, backend="rust")
    np.testing.assert_array_equal(actual, expected)
    assert events == expected_events

    monkeypatch.setattr(backends, "_HAS_RUST", False)
    python_auto, python_expected = DPINeuron(), DPINeuron()
    actual, events = python_auto.simulate(100, 5.0)
    expected, expected_events = python_expected.simulate(100, 5.0, backend="python")
    np.testing.assert_array_equal(actual, expected)
    assert events == expected_events
