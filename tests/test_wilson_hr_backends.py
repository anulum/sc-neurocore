# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Polyglot parity tests for the Wilson 1999 cortical model

"""Cross-backend parity for ``WilsonHRNeuron.simulate``.

Wilson's polynomial cortical model is a two-dimensional autonomous RK4 flow with
an exact polynomial right-hand side (no transcendental functions) plus a hard
voltage reset on threshold, so Rust, Julia and Go reproduce the NumPy reference
**bit-for-bit**. Mojo's release build contracts the RK4 multiply-adds into fused
multiply-adds (one rounding instead of two); the per-spike hard reset re-anchors
the trajectory and a two-dimensional autonomous flow cannot be chaotic, so the
single-ULP difference does not accumulate — the whole-trace gap stays tight and
the spike counts match.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models import wilson_hr
from sc_neurocore.neurons.models.wilson_hr import WilsonHRNeuron

_ULP = float(np.spacing(1.0))
_STEP_TOL = 8.0 * _ULP


def _run(backend: str, *, n: int = 4000, current: float = 5.0, **params) -> tuple:
    neuron = WilsonHRNeuron(**params)
    trace, spikes = neuron.simulate(n, current, backend=backend)
    return trace, spikes, neuron.v, neuron.r


def _rust() -> bool:
    return wilson_hr._HAS_RUST


def _julia() -> bool:
    return wilson_hr._ensure_julia_loaded()


def _go() -> bool:
    return wilson_hr._ensure_go_loaded()


def _mojo() -> bool:
    return wilson_hr._ensure_mojo_loaded()


_BIT_EXACT = [("rust", _rust), ("julia", _julia), ("go", _go)]
# Mix of subthreshold (0.3, 1.0) and spiking (5.0, 10.0) drives.
_CURRENTS = [0.3, 1.0, 5.0, 10.0]
_REGIMES = [
    dict(tau_r=1.9, dt=0.05),
    dict(tau_r=2.5, dt=0.04),
    dict(tau_r=1.2, dt=0.05, v_peak=0.35),
]


# ───────────────────── bit-exact backends (rust/julia/go) ─────────────────────


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
@pytest.mark.parametrize("current", _CURRENTS)
def test_bit_exact_trace_currents(backend: str, available, current: float) -> None:
    if not available():
        pytest.skip(f"{backend} Wilson-HR backend unavailable")
    ref_trace, ref_spikes, rv, rr = _run("python", current=current)
    trace, spikes, vf, rf = _run(backend, current=current)
    np.testing.assert_array_equal(trace, ref_trace)
    assert spikes == ref_spikes
    assert vf == rv and rf == rr


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
@pytest.mark.parametrize("regime", _REGIMES, ids=["tau19", "tau25", "tau12"])
def test_bit_exact_trace_regimes(backend: str, available, regime: dict) -> None:
    if not available():
        pytest.skip(f"{backend} Wilson-HR backend unavailable")
    ref_trace, ref_spikes, rv, rr = _run("python", current=10.0, **regime)
    trace, spikes, vf, rf = _run(backend, current=10.0, **regime)
    np.testing.assert_array_equal(trace, ref_trace)
    assert spikes == ref_spikes
    assert vf == rv and rf == rr


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
def test_bit_exact_empty_and_single(backend: str, available) -> None:
    if not available():
        pytest.skip(f"{backend} Wilson-HR backend unavailable")
    for n in (0, 1, 2):
        ref, rs, rv, rr = _run("python", n=n)
        got, gs, gv, gr = _run(backend, n=n)
        np.testing.assert_array_equal(got, ref)
        assert (gs, gv, gr) == (rs, rv, rr)


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
def test_bit_exact_long_horizon(backend: str, available) -> None:
    if not available():
        pytest.skip(f"{backend} Wilson-HR backend unavailable")
    ref, rs, rv, rr = _run("python", n=60_000, current=10.0)
    got, gs, gv, gr = _run(backend, n=60_000, current=10.0)
    np.testing.assert_array_equal(got, ref)
    assert (gs, gv, gr) == (rs, rv, rr)


# ───────────────────────────── Mojo (FMA, non-amplifying) ─────────────────────


@pytest.mark.skipif(not _mojo(), reason="Mojo Wilson-HR backend unavailable")
@pytest.mark.parametrize("current", _CURRENTS)
def test_mojo_whole_trace_non_amplifying(current: float) -> None:
    # The hard reset re-anchors the trajectory and a 2D autonomous flow cannot be
    # chaotic, so the FMA ULP does not accumulate: the whole trace stays allclose
    # and the spike count matches exactly.
    ref, ref_spikes, _rv, _rr = _run("python", n=20_000, current=current)
    got, spikes, _vf, _rf = _run("mojo", n=20_000, current=current)
    np.testing.assert_allclose(got, ref, atol=1e-9, rtol=0.0)
    assert spikes == ref_spikes


@pytest.mark.skipif(not _mojo(), reason="Mojo Wilson-HR backend unavailable")
def test_mojo_per_step_within_tolerance() -> None:
    rng = np.random.default_rng(99)
    worst = 0.0
    for _ in range(5000):
        v = float(rng.uniform(-0.8, 0.39))
        r = float(rng.uniform(-0.2, 0.6))
        cur = float(rng.uniform(0.0, 10.0))
        ref, _rs, rv, rr = WilsonHRNeuron(v=v, r=r)._simulate_python(1, cur)
        got, _gs, gv, gr = WilsonHRNeuron(v=v, r=r)._simulate_mojo(1, cur)
        worst = max(worst, abs(ref[0] - got[0]), abs(rv - gv), abs(rr - gr))
    assert worst <= _STEP_TOL, f"per-step Mojo gap {worst} exceeds {_STEP_TOL}"


# ───────────────────────────── dispatch + algorithm ───────────────────────────


def test_auto_matches_python() -> None:
    ref, ref_spikes, _rv, _rr = _run("python")
    got, spikes, _vf, _rf = _run("auto")
    np.testing.assert_allclose(got, ref, atol=1e-9, rtol=0.0)
    assert spikes == ref_spikes


def test_invalid_backend_raises() -> None:
    with pytest.raises(ValueError, match="backend must be"):
        WilsonHRNeuron().simulate(10, 0.0, backend="cuda")


def test_negative_n_steps_raises() -> None:
    with pytest.raises(ValueError, match="n_steps must be non-negative"):
        WilsonHRNeuron().simulate(-1, 0.0)


def test_simulate_matches_repeated_step() -> None:
    trace_a, spikes_a = WilsonHRNeuron().simulate(300, 10.0, backend="python")
    manual = []
    spikes_b = 0
    stepper = WilsonHRNeuron()
    for _ in range(300):
        spikes_b += stepper.step(10.0)
        manual.append(stepper.v)
    np.testing.assert_array_equal(trace_a, np.asarray(manual, dtype=np.float64))
    assert spikes_a == spikes_b


def test_hard_reset_recorded_in_trace() -> None:
    # On a spiking step the recorded sample is the post-reset -0.7, exactly as the
    # per-step path leaves it.
    trace, spikes = WilsonHRNeuron().simulate(20_000, 10.0, backend="python")
    if spikes:
        assert np.any(trace == -0.7)
