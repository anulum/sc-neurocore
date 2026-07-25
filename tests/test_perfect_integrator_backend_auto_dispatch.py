# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Perfect Integrator automatic backend dispatch

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.accel import perfect_integrator as backends
from sc_neurocore.neurons.models.perfect_integrator import PerfectIntegratorNeuron
from tests.perfect_integrator_backends_support import configured_neuron


def test_auto_prefers_measured_first_mojo() -> None:
    """Route a non-default instance through measured-first Mojo."""
    auto = configured_neuron()
    expected = configured_neuron()
    auto_trace, auto_spikes = auto.simulate(100, 2.2, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 2.2, backend="mojo")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.v) == (expected_spikes, expected.v)


def test_auto_falls_through_to_julia(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use Julia when the measured-first Mojo lane is unavailable."""
    monkeypatch.setattr(backends, "ensure_mojo_loaded", lambda: False)
    auto = configured_neuron()
    expected = configured_neuron()
    auto_trace, auto_spikes = auto.simulate(100, 2.2, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 2.2, backend="julia")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.v) == (expected_spikes, expected.v)


def test_auto_falls_through_to_go(monkeypatch: pytest.MonkeyPatch) -> None:
    """Continue to Go when Mojo and Julia are unavailable."""
    monkeypatch.setattr(backends, "ensure_mojo_loaded", lambda: False)
    monkeypatch.setattr(backends, "ensure_julia_loaded", lambda: False)
    auto = configured_neuron()
    expected = configured_neuron()
    auto_trace, auto_spikes = auto.simulate(100, 2.2, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 2.2, backend="go")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.v) == (expected_spikes, expected.v)


def test_auto_falls_through_to_factory_rust(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use Rust when every full-parameter compiled lane is unavailable."""
    monkeypatch.setattr(backends, "ensure_mojo_loaded", lambda: False)
    monkeypatch.setattr(backends, "ensure_julia_loaded", lambda: False)
    monkeypatch.setattr(backends, "ensure_go_loaded", lambda: False)
    auto = PerfectIntegratorNeuron()
    expected = PerfectIntegratorNeuron()
    auto_trace, auto_spikes = auto.simulate(100, 5.0, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 5.0, backend="rust")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.v) == (expected_spikes, expected.v)


def test_auto_falls_back_to_python(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retain the Python floor when no compatible compiled lane is available."""
    monkeypatch.setattr(backends, "ensure_mojo_loaded", lambda: False)
    monkeypatch.setattr(backends, "ensure_julia_loaded", lambda: False)
    monkeypatch.setattr(backends, "ensure_go_loaded", lambda: False)
    monkeypatch.setattr(backends, "_HAS_RUST", False)
    auto = PerfectIntegratorNeuron()
    expected = PerfectIntegratorNeuron()
    auto_trace, auto_spikes = auto.simulate(100, 5.0, backend="auto")
    expected_trace, expected_spikes = expected.simulate(100, 5.0, backend="python")
    np.testing.assert_array_equal(auto_trace, expected_trace)
    assert (auto_spikes, auto.v) == (expected_spikes, expected.v)
