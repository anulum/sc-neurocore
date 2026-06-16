# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Polyglot parity tests for the Mihalas-Niebur 2009 generalised IF

"""Cross-backend parity for ``MihalasNieburNeuron.simulate``.

The Mihalas-Niebur 2009 generalised integrate-and-fire model has a purely linear
right-hand side (additions, multiplications, divisions; no transcendental
functions) advanced by candidate-first RK4 with a discontinuous spike reset.
Because every stage is exact arithmetic, the Rust engine, Julia and Go backends
reproduce the pure-NumPy reference bit-for-bit — trace, spike count and the final
``(v, theta, i1, i2)`` state all match exactly. The Mojo backend fuses
multiply-add, so it is validated as non-amplifying within a tight ULP band with
identical spike counts rather than bit-exactly (the band is compiler/version
dependent and must not be promoted to a strict equality assertion).
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models import mihalas_niebur
from sc_neurocore.neurons.models.mihalas_niebur import MihalasNieburNeuron


def _run(backend: str, *, n: int = 4000, current: float = 2.0, **params) -> tuple:
    neuron = MihalasNieburNeuron(**params)
    trace, spikes = neuron.simulate(n, current, backend=backend)
    return trace, spikes, (neuron.v, neuron.theta, neuron.i1, neuron.i2)


def _rust() -> bool:
    return mihalas_niebur._HAS_RUST


def _julia() -> bool:
    return mihalas_niebur._ensure_julia_loaded()


def _go() -> bool:
    return mihalas_niebur._ensure_go_loaded()


def _mojo() -> bool:
    return mihalas_niebur._ensure_mojo_loaded()


_BIT_EXACT = [("rust", _rust), ("julia", _julia), ("go", _go)]
_CURRENTS = [0.0, 1.5, 2.0, 3.0]
_REGIMES = [
    dict(),
    dict(a=0.5, b=0.05, r1=0.5, r2=0.3),
    dict(a=1.0, b=0.1, theta_inf=0.8, r1=1.0, r2=0.2),
    dict(tau_v=5.0, tau_theta=50.0, a=0.3, b=0.2),
]


# ───────────── Rust / Julia / Go: bit-exact (linear exact arithmetic) ──────────


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
@pytest.mark.parametrize("current", _CURRENTS)
def test_bit_exact_currents(backend: str, available, current: float) -> None:
    if not available():
        pytest.skip(f"{backend} Mihalas-Niebur backend unavailable")
    ref_trace, ref_spikes, ref_state = _run("python", current=current)
    trace, spikes, state = _run(backend, current=current)
    np.testing.assert_array_equal(trace, ref_trace)
    assert spikes == ref_spikes
    assert state == ref_state


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
@pytest.mark.parametrize("regime", _REGIMES, ids=["r0", "r1", "r2", "r3"])
def test_bit_exact_regimes(backend: str, available, regime: dict) -> None:
    if not available():
        pytest.skip(f"{backend} Mihalas-Niebur backend unavailable")
    ref_trace, ref_spikes, ref_state = _run("python", current=2.0, **regime)
    trace, spikes, state = _run(backend, current=2.0, **regime)
    np.testing.assert_array_equal(trace, ref_trace)
    assert spikes == ref_spikes
    assert state == ref_state


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
def test_bit_exact_empty_and_long_horizon(backend: str, available) -> None:
    if not available():
        pytest.skip(f"{backend} Mihalas-Niebur backend unavailable")
    for n in (0, 1, 2, 60_000):
        ref, rs, rstate = _run("python", n=n)
        got, gs, gstate = _run(backend, n=n)
        np.testing.assert_array_equal(got, ref)
        assert (gs, gstate) == (rs, rstate)


# ───────────────────── Mojo: non-amplifying (FMA fusion) ───────────────────────


@pytest.mark.parametrize("current", _CURRENTS)
def test_mojo_non_amplifying_currents(current: float) -> None:
    if not _mojo():
        pytest.skip("mojo Mihalas-Niebur backend unavailable")
    ref, ref_spikes, _state = _run("python", n=20_000, current=current)
    got, spikes, _gstate = _run("mojo", n=20_000, current=current)
    np.testing.assert_allclose(got, ref, atol=1e-9, rtol=0.0)
    assert spikes == ref_spikes


@pytest.mark.parametrize("regime", _REGIMES, ids=["r0", "r1", "r2", "r3"])
def test_mojo_non_amplifying_regimes(regime: dict) -> None:
    if not _mojo():
        pytest.skip("mojo Mihalas-Niebur backend unavailable")
    # The gap at 60k steps must remain at the ULP level (no amplification).
    ref, ref_spikes, _state = _run("python", n=60_000, current=2.0, **regime)
    got, spikes, _gstate = _run("mojo", n=60_000, current=2.0, **regime)
    assert float(np.max(np.abs(got - ref))) < 1e-8
    assert spikes == ref_spikes


# ───────────────────────────── dispatch + algorithm ───────────────────────────


def test_auto_matches_python() -> None:
    ref, ref_spikes, ref_state = _run("python")
    got, spikes, state = _run("auto")
    np.testing.assert_allclose(got, ref, atol=1e-9, rtol=0.0)
    assert spikes == ref_spikes
    assert state == ref_state


def test_invalid_backend_raises() -> None:
    with pytest.raises(ValueError, match="backend must be"):
        MihalasNieburNeuron().simulate(10, 0.0, backend="cuda")


def test_negative_n_steps_raises() -> None:
    with pytest.raises(ValueError, match="n_steps must be non-negative"):
        MihalasNieburNeuron().simulate(-1, 0.0)


def test_non_finite_current_raises() -> None:
    with pytest.raises(ValueError, match="current must be finite"):
        MihalasNieburNeuron().simulate(10, float("inf"))


def test_invalid_runtime_raises() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        MihalasNieburNeuron(tau_v=-1.0).simulate(10, 2.0)


def test_simulate_matches_repeated_step() -> None:
    trace_a, spikes_a = MihalasNieburNeuron().simulate(400, 2.0, backend="python")
    manual = []
    spikes_b = 0
    stepper = MihalasNieburNeuron()
    for _ in range(400):
        spikes_b += stepper.step(2.0)
        manual.append(stepper.v)
    np.testing.assert_array_equal(trace_a, np.asarray(manual, dtype=np.float64))
    assert spikes_a == spikes_b
