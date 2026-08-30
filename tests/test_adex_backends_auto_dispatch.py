# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (auto_dispatch) from former test_adex_backends.py

from __future__ import annotations

from tests.adex_backends_support import *  # noqa: F403


def test_auto_prefers_measured_fastest_backend() -> None:
    """Route baseline Euler through Rust before slower compiled lanes."""
    assert adex._HAS_RUST, "Rust AdEx backend must be built for the fidelity gate"
    auto = AdExNeuron(v=-60.0, w=3.0)
    expected = AdExNeuron(v=-60.0, w=3.0)
    auto_trace, auto_spikes = auto.simulate(100, 250.0, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 250.0, backend="rust")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.v, auto.w) == (expected_spikes, expected.v, expected.w)


def test_auto_falls_through_to_go_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Continue through measured order when Rust and Julia are unavailable."""
    _require_adex_backend("go")
    monkeypatch.setattr(adex, "_HAS_RUST", False)
    monkeypatch.setattr(adex, "_ensure_julia_loaded", lambda: False)
    auto = AdExNeuron(v=-60.0, w=3.0)
    expected = AdExNeuron(v=-60.0, w=3.0)
    auto_trace, auto_spikes = auto.simulate(100, 250.0, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 250.0, backend="go")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.v, auto.w) == (expected_spikes, expected.v, expected.w)


def test_auto_falls_through_to_julia(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use Julia when the measured Rust lane is unavailable."""
    _require_adex_backend("julia")
    monkeypatch.setattr(adex, "_HAS_RUST", False)
    auto = AdExNeuron(v=-60.0, w=3.0)
    expected = AdExNeuron(v=-60.0, w=3.0)
    auto_trace, auto_spikes = auto.simulate(100, 250.0, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 250.0, backend="julia")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.v, auto.w) == (expected_spikes, expected.v, expected.w)


def test_auto_falls_through_to_mojo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use Mojo after the faster full-parameter lanes are unavailable."""
    _require_adex_backend("mojo")
    monkeypatch.setattr(adex, "_HAS_RUST", False)
    monkeypatch.setattr(adex, "_ensure_julia_loaded", lambda: False)
    monkeypatch.setattr(adex, "_ensure_go_loaded", lambda: False)
    auto = AdExNeuron()
    expected = AdExNeuron()
    auto_trace, auto_spikes = auto.simulate(100, 250.0, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 250.0, backend="mojo")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.v, auto.w) == (expected_spikes, expected.v, expected.w)


def test_auto_falls_back_to_python(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retain the Python floor when no compatible compiled lane is available."""
    monkeypatch.setattr(adex, "_ensure_julia_loaded", lambda: False)
    monkeypatch.setattr(adex, "_ensure_mojo_loaded", lambda: False)
    monkeypatch.setattr(adex, "_ensure_go_loaded", lambda: False)
    monkeypatch.setattr(adex, "_HAS_RUST", False)
    auto = AdExNeuron()
    expected = AdExNeuron()
    auto_trace, auto_spikes = auto.simulate(100, 250.0, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 250.0, backend="python")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.v, auto.w) == (expected_spikes, expected.v, expected.w)
