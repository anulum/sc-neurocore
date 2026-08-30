# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ExpIF complete-packet and source-profile contracts

from __future__ import annotations

import math

import numpy as np
import pytest

import sc_neurocore.neurons.models.expif as expif
from sc_neurocore.neurons.models.expif import ExpIFNeuron


def test_python_complete_packet_is_aligned_and_committed_once() -> None:
    neuron = ExpIFNeuron(refractory_period=0.06)
    voltage, refractory, events = neuron.simulate_complete(2_000, 50.0, backend="python")

    assert voltage.shape == refractory.shape == events.shape == (2_000,)
    assert voltage.flags.c_contiguous
    assert refractory.flags.c_contiguous
    assert events.dtype == np.uint8
    assert set(np.unique(events)).issubset({0, 1})
    assert int(events.sum()) > 0
    assert neuron.v == voltage[-1]
    assert neuron.refractory_remaining == refractory[-1]


def test_python_batch_is_atomic_on_a_late_numeric_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    neuron = ExpIFNeuron(v=-62.0)
    before = (neuron.v, neuron.refractory_remaining)
    original = ExpIFNeuron._rk4_candidate
    calls = 0

    def fail_on_third(candidate: ExpIFNeuron, current: float) -> float:
        nonlocal calls
        calls += 1
        if calls == 3:
            return math.inf
        return original(candidate, current)

    monkeypatch.setattr(ExpIFNeuron, "_rk4_candidate", fail_on_third)
    with pytest.raises(ValueError, match="ExpIF update"):
        neuron.simulate_complete(5, 5.0, backend="python")
    assert calls == 3
    assert (neuron.v, neuron.refractory_remaining) == before


def test_source_factory_freezes_the_paper_fit_and_numerical_specialization() -> None:
    source = ExpIFNeuron.fourcaud_trocme_2003()

    assert source.profile == "fourcaud_trocme_2003"
    assert source.v_threshold == -30.0
    assert source.dt == 0.01
    assert source.refractory_period == 1.7
    assert source.analytical_tail_ms() == pytest.approx(0.001855930799631619, abs=1.0e-15)


def test_source_profile_rejects_non_source_fit_and_timestep() -> None:
    with pytest.raises(ValueError, match="fitted source values"):
        ExpIFNeuron(profile="fourcaud_trocme_2003")
    with pytest.raises(ValueError, match="dt < 0.02"):
        ExpIFNeuron.fourcaud_trocme_2003(dt=0.02)


def test_source_profile_has_substep_convergence() -> None:
    coarse = ExpIFNeuron.fourcaud_trocme_2003(dt=0.01)
    fine = ExpIFNeuron.fourcaud_trocme_2003(dt=0.005)
    coarse_trace, _, coarse_events = coarse.simulate_complete(2_000, 20.0, backend="python")
    fine_trace, _, fine_events = fine.simulate_complete(4_000, 20.0, backend="python")

    assert int(coarse_events.sum()) == int(fine_events.sum())
    assert coarse.v == pytest.approx(fine.v, abs=2.0e-2)
    assert np.all(np.isfinite(coarse_trace))
    assert np.all(np.isfinite(fine_trace))


@pytest.mark.skipif(not expif._HAS_RUST, reason="current ExpIF batch binding is unavailable")
def test_source_profile_matches_production_rust_complete_packet() -> None:
    python = ExpIFNeuron.fourcaud_trocme_2003()
    rust = ExpIFNeuron.fourcaud_trocme_2003()
    expected = python.simulate_complete(4_000, 20.0, backend="python")
    actual = rust.simulate_complete(4_000, 20.0, backend="rust")

    np.testing.assert_allclose(actual[0], expected[0], rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(actual[1], expected[1], rtol=0.0, atol=1.0e-12)
    np.testing.assert_array_equal(actual[2], expected[2])


@pytest.mark.parametrize("backend", ("julia", "go", "mojo"))
def test_source_profile_matches_each_native_complete_packet(backend: str) -> None:
    loader = getattr(expif, f"_ensure_{backend}_loaded")
    if not loader():
        pytest.skip(f"{backend} ExpIF backend is unavailable")
    python = ExpIFNeuron.fourcaud_trocme_2003()
    native = ExpIFNeuron.fourcaud_trocme_2003()
    expected = python.simulate_complete(4_000, 20.0, backend="python")
    actual = native.simulate_complete(4_000, 20.0, backend=backend)

    np.testing.assert_allclose(actual[0], expected[0], rtol=0.0, atol=5.0e-8)
    np.testing.assert_allclose(actual[1], expected[1], rtol=0.0, atol=1.0e-12)
    np.testing.assert_array_equal(actual[2], expected[2])
