# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (auto_dispatch) from former test_expif_backends.py

from __future__ import annotations

from tests.expif_backends_support import *  # noqa: F403


def test_auto_prefers_first_full_parameter_backend() -> None:
    """Route a non-default instance through measured-first Julia."""
    _require_expif_backend("julia")
    auto = _configured()
    expected = _configured()
    auto_trace, auto_spikes = auto.simulate(100, 50.0, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 50.0, backend="julia")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.v, auto.refractory_remaining) == (
        expected_spikes,
        expected.v,
        expected.refractory_remaining,
    )


def test_auto_falls_through_to_go(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use Go when the measured-first Julia lane is unavailable."""
    _require_expif_backend("go")
    monkeypatch.setattr(expif, "_ensure_julia_loaded", lambda: False)
    auto = _configured()
    expected = _configured()
    auto_trace, auto_spikes = auto.simulate(100, 50.0, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 50.0, backend="go")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.v, auto.refractory_remaining) == (
        expected_spikes,
        expected.v,
        expected.refractory_remaining,
    )


def test_auto_falls_through_to_mojo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Continue to Mojo when Julia and Go are unavailable."""
    _require_expif_backend("mojo")
    monkeypatch.setattr(expif, "_ensure_julia_loaded", lambda: False)
    monkeypatch.setattr(expif, "_ensure_go_loaded", lambda: False)
    auto = _configured()
    expected = _configured()
    auto_trace, auto_spikes = auto.simulate(100, 50.0, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 50.0, backend="mojo")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.v, auto.refractory_remaining) == (
        expected_spikes,
        expected.v,
        expected.refractory_remaining,
    )


def test_auto_falls_through_to_factory_rust(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use Rust when every full-parameter compiled lane is unavailable."""
    monkeypatch.setattr(expif, "_ensure_mojo_loaded", lambda: False)
    monkeypatch.setattr(expif, "_ensure_julia_loaded", lambda: False)
    monkeypatch.setattr(expif, "_ensure_go_loaded", lambda: False)
    auto = ExpIFNeuron()
    expected = ExpIFNeuron()
    auto_trace, auto_spikes = auto.simulate(100, 20.0, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 20.0, backend="rust")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.v, auto.refractory_remaining) == (
        expected_spikes,
        expected.v,
        expected.refractory_remaining,
    )


def test_auto_falls_back_to_python(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retain the Python floor when no compatible compiled lane is available."""
    monkeypatch.setattr(expif, "_ensure_mojo_loaded", lambda: False)
    monkeypatch.setattr(expif, "_ensure_julia_loaded", lambda: False)
    monkeypatch.setattr(expif, "_ensure_go_loaded", lambda: False)
    monkeypatch.setattr(expif, "_HAS_RUST", False)
    auto = ExpIFNeuron()
    expected = ExpIFNeuron()
    auto_trace, auto_spikes = auto.simulate(100, 20.0, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 20.0, backend="python")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.v, auto.refractory_remaining) == (
        expected_spikes,
        expected.v,
        expected.refractory_remaining,
    )
