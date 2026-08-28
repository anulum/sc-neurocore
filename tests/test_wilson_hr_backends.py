# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Polyglot parity tests for the Wilson 1999 cortical model

"""Cross-backend parity for ``WilsonHRNeuron.simulate``.

Wilson's polynomial cortical model is a continuous two-dimensional autonomous
RK4 flow with an exact polynomial right-hand side. Rust, Julia, and Go reproduce
the NumPy reference bit-for-bit. Mojo is checked at one-step precision and over
a bounded complete trajectory; no reset is present to hide phase drift.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from sc_neurocore.neurons.models import wilson_hr
from sc_neurocore.neurons.models.wilson_hr import WilsonHRNeuron

_ULP = float(np.spacing(1.0))
_STEP_TOL = 8.0 * _ULP
BackendAvailable = Callable[[], bool]
RunResult = tuple[NDArray[np.float64], int, float, float]


def _run(
    backend: str,
    *,
    n: int = 4000,
    current: float = 0.1,
    tau_r: float = 1.9,
    dt: float = 0.05,
    v_peak: float = 0.0,
) -> RunResult:
    neuron = WilsonHRNeuron(tau_r=tau_r, dt=dt, v_peak=v_peak)
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


_BIT_EXACT: tuple[tuple[str, BackendAvailable], ...] = (
    ("rust", _rust),
    ("julia", _julia),
    ("go", _go),
)
# Mix of subthreshold and periodic source regimes.
_CURRENTS = [0.0, 0.03, 0.075, 0.14]
_REGIMES: tuple[dict[str, float], ...] = (
    dict(tau_r=1.9, dt=0.05),
    dict(tau_r=2.5, dt=0.04),
    dict(tau_r=1.2, dt=0.05, v_peak=-0.1),
)


# ───────────────────── bit-exact backends (rust/julia/go) ─────────────────────


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
@pytest.mark.parametrize("current", _CURRENTS)
def test_bit_exact_trace_currents(
    backend: str, available: BackendAvailable, current: float
) -> None:
    if not available():
        pytest.skip(f"{backend} Wilson-HR backend unavailable")
    ref_trace, ref_spikes, rv, rr = _run("python", current=current)
    trace, spikes, vf, rf = _run(backend, current=current)
    np.testing.assert_array_equal(trace, ref_trace)
    assert spikes == ref_spikes
    assert vf == rv and rf == rr


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
@pytest.mark.parametrize("regime", _REGIMES, ids=["tau19", "tau25", "tau12"])
def test_bit_exact_trace_regimes(
    backend: str, available: BackendAvailable, regime: dict[str, float]
) -> None:
    if not available():
        pytest.skip(f"{backend} Wilson-HR backend unavailable")
    tau_r = regime.get("tau_r", 1.9)
    dt = regime.get("dt", 0.05)
    v_peak = regime.get("v_peak", 0.0)
    ref_trace, ref_spikes, rv, rr = _run("python", current=0.1, tau_r=tau_r, dt=dt, v_peak=v_peak)
    trace, spikes, vf, rf = _run(backend, current=0.1, tau_r=tau_r, dt=dt, v_peak=v_peak)
    np.testing.assert_array_equal(trace, ref_trace)
    assert spikes == ref_spikes
    assert vf == rv and rf == rr


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
def test_bit_exact_empty_and_single(backend: str, available: BackendAvailable) -> None:
    if not available():
        pytest.skip(f"{backend} Wilson-HR backend unavailable")
    for n in (0, 1, 2):
        ref, rs, rv, rr = _run("python", n=n)
        got, gs, gv, gr = _run(backend, n=n)
        np.testing.assert_array_equal(got, ref)
        assert (gs, gv, gr) == (rs, rv, rr)


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
def test_bit_exact_long_horizon(backend: str, available: BackendAvailable) -> None:
    if not available():
        pytest.skip(f"{backend} Wilson-HR backend unavailable")
    ref, rs, rv, rr = _run("python", n=60_000, current=0.1)
    got, gs, gv, gr = _run(backend, n=60_000, current=0.1)
    np.testing.assert_array_equal(got, ref)
    assert (gs, gv, gr) == (rs, rv, rr)


# ───────────────────────────── Mojo (FMA, non-amplifying) ─────────────────────


@pytest.mark.skipif(not _mojo(), reason="Mojo Wilson-HR backend unavailable")
@pytest.mark.parametrize("current", _CURRENTS)
def test_mojo_whole_trace_non_amplifying(current: float) -> None:
    ref, ref_spikes, _rv, _rr = _run("python", n=4_000, current=current)
    got, spikes, _vf, _rf = _run("mojo", n=4_000, current=current)
    np.testing.assert_allclose(got, ref, atol=1e-9, rtol=0.0)
    assert spikes == ref_spikes


@pytest.mark.skipif(not _mojo(), reason="Mojo Wilson-HR backend unavailable")
def test_mojo_per_step_within_tolerance() -> None:
    rng = np.random.default_rng(99)
    worst = 0.0
    for _ in range(5000):
        v = float(rng.uniform(-0.8, 0.35))
        r = float(rng.uniform(-0.2, 0.6))
        cur = float(rng.uniform(0.0, 0.2))
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
    trace_a, spikes_a = WilsonHRNeuron().simulate(300, 0.1, backend="python")
    manual = []
    spikes_b = 0
    stepper = WilsonHRNeuron()
    for _ in range(300):
        spikes_b += stepper.step(0.1)
        manual.append(stepper.v)
    np.testing.assert_array_equal(trace_a, np.asarray(manual, dtype=np.float64))
    assert spikes_a == spikes_b


def test_spike_samples_preserve_continuous_trace() -> None:
    trace, spikes = WilsonHRNeuron().simulate(20_000, 0.1, backend="python")
    assert spikes > 0
    assert np.any(trace > 0.0)
    assert not np.any(trace == -0.7)
