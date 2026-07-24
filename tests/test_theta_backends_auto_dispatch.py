# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (auto_dispatch) from former test_theta_backends.py

from __future__ import annotations

from tests.theta_backends_support import *  # noqa: F403


def test_auto_prefers_go_without_initialising_other_runtimes() -> None:
    """Route through Go without initialising Julia or probing Mojo."""
    with (
        patch.object(backends, "ensure_julia_loaded") as ensure_julia,
        patch.object(backends, "ensure_mojo_loaded") as ensure_mojo,
    ):
        auto = _configured()
        expected = _configured()
        auto_trace, auto_spikes = auto.simulate(100, 2.2, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 2.2, backend="go")
    ensure_julia.assert_not_called()
    ensure_mojo.assert_not_called()
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.theta) == (expected_spikes, expected.theta)


def test_auto_falls_through_to_julia(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use Julia when the Go shared library is unavailable."""
    monkeypatch.setattr(backends, "ensure_go_loaded", lambda: False)
    auto = _configured()
    expected = _configured()
    auto_trace, auto_spikes = auto.simulate(100, 2.2, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 2.2, backend="julia")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.theta) == (expected_spikes, expected.theta)


def test_auto_falls_through_to_mojo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Continue to Mojo when Go and Julia are unavailable."""
    monkeypatch.setattr(backends, "ensure_go_loaded", lambda: False)
    monkeypatch.setattr(backends, "ensure_julia_loaded", lambda: False)
    auto = _configured()
    expected = _configured()
    auto_trace, auto_spikes = auto.simulate(100, 2.2, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 2.2, backend="mojo")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.theta) == (expected_spikes, expected.theta)


def test_auto_falls_through_to_factory_rust(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use Rust when every full-parameter compiled lane is unavailable."""
    monkeypatch.setattr(backends, "ensure_mojo_loaded", lambda: False)
    monkeypatch.setattr(backends, "ensure_julia_loaded", lambda: False)
    monkeypatch.setattr(backends, "ensure_go_loaded", lambda: False)
    auto = ThetaNeuron()
    expected = ThetaNeuron()
    auto_trace, auto_spikes = auto.simulate(100, 5.0, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 5.0, backend="rust")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.theta) == (expected_spikes, expected.theta)


def test_auto_falls_back_to_python(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retain the Python floor when no compatible compiled lane is available."""
    monkeypatch.setattr(backends, "ensure_mojo_loaded", lambda: False)
    monkeypatch.setattr(backends, "ensure_julia_loaded", lambda: False)
    monkeypatch.setattr(backends, "ensure_go_loaded", lambda: False)
    monkeypatch.setattr(backends, "_HAS_RUST", False)
    auto = ThetaNeuron()
    expected = ThetaNeuron()
    auto_trace, auto_spikes = auto.simulate(100, 5.0, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 5.0, backend="python")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.theta) == (expected_spikes, expected.theta)
