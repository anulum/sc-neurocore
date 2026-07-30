# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Polyglot parity tests for the retained SC triangular recurrence

"""Cross-backend parity for ``McKeanNeuron.simulate``.

McKean's caricature is a two-dimensional autonomous RK4 flow with a
piecewise-linear right-hand side: exact floating-point arithmetic (additions,
multiplications and branch selection — no transcendental functions), so Rust,
Julia and Go reproduce the NumPy reference **bit-for-bit**. Mojo's release build
contracts the RK4 multiply-adds into fused multiply-adds (one rounding instead of
two); because a two-dimensional autonomous flow cannot be chaotic, that single-ULP
difference does not amplify — the whole-trace gap stays within a tight band over
long horizons and the spike counts match.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models import sc_triangular_mckean as mckean
from sc_neurocore.neurons.models.sc_triangular_mckean import (
    SCTriangularMcKeanNeuron as McKeanNeuron,
)

_ULP = float(np.spacing(1.0))
_STEP_TOL = 8.0 * _ULP


def _run(backend: str, *, n: int = 4000, current: float = 0.5, **params) -> tuple:
    neuron = McKeanNeuron(**params)
    trace, spikes = neuron.simulate(n, current, backend=backend)
    return trace, spikes, neuron.v, neuron.w


def _rust() -> bool:
    return mckean._HAS_RUST


def _julia() -> bool:
    return mckean._ensure_julia_loaded()


def _go() -> bool:
    return mckean._ensure_go_loaded()


def _mojo() -> bool:
    return mckean._ensure_mojo_loaded()


_BIT_EXACT = [("rust", _rust), ("julia", _julia), ("go", _go)]
_CURRENTS = [0.3, 0.5, 0.8, 1.0]
_REGIMES = [
    dict(a=0.25, epsilon=0.01, gamma=0.5),
    dict(a=0.1, epsilon=0.02, gamma=0.4),
    dict(a=0.4, epsilon=0.005, gamma=0.6),
]


# ───────────────────── bit-exact backends (rust/julia/go) ─────────────────────


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
@pytest.mark.parametrize("current", _CURRENTS)
def test_bit_exact_trace_currents(backend: str, available, current: float) -> None:
    if not available():
        pytest.skip(f"{backend} SC triangular backend unavailable")
    ref_trace, ref_spikes, rv, rw = _run("python", current=current)
    trace, spikes, vf, wf = _run(backend, current=current)
    np.testing.assert_array_equal(trace, ref_trace)
    assert spikes == ref_spikes
    assert vf == rv and wf == rw


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
@pytest.mark.parametrize("regime", _REGIMES, ids=["a25", "a10", "a40"])
def test_bit_exact_trace_regimes(backend: str, available, regime: dict) -> None:
    if not available():
        pytest.skip(f"{backend} McKean backend unavailable")
    ref_trace, ref_spikes, rv, rw = _run("python", current=0.6, **regime)
    trace, spikes, vf, wf = _run(backend, current=0.6, **regime)
    np.testing.assert_array_equal(trace, ref_trace)
    assert spikes == ref_spikes
    assert vf == rv and wf == rw


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
def test_bit_exact_empty_and_single(backend: str, available) -> None:
    if not available():
        pytest.skip(f"{backend} McKean backend unavailable")
    for n in (0, 1, 2):
        ref, rs, rv, rw = _run("python", n=n)
        got, gs, gv, gw = _run(backend, n=n)
        np.testing.assert_array_equal(got, ref)
        assert (gs, gv, gw) == (rs, rv, rw)


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
def test_bit_exact_long_horizon(backend: str, available) -> None:
    if not available():
        pytest.skip(f"{backend} McKean backend unavailable")
    ref, rs, rv, rw = _run("python", n=60_000, current=0.5)
    got, gs, gv, gw = _run(backend, n=60_000, current=0.5)
    np.testing.assert_array_equal(got, ref)
    assert (gs, gv, gw) == (rs, rv, rw)


# ───────────────────────────── Mojo (FMA, non-amplifying) ─────────────────────


@pytest.mark.skipif(not _mojo(), reason="Mojo McKean backend unavailable")
@pytest.mark.parametrize("current", _CURRENTS)
def test_mojo_whole_trace_non_amplifying(current: float) -> None:
    # A 2D autonomous flow cannot be chaotic, so the FMA ULP does not amplify:
    # the whole trace stays allclose and the spike count matches exactly.
    ref, ref_spikes, _rv, _rw = _run("python", n=20_000, current=current)
    got, spikes, _vf, _wf = _run("mojo", n=20_000, current=current)
    np.testing.assert_allclose(got, ref, atol=1e-9, rtol=0.0)
    assert spikes == ref_spikes


@pytest.mark.skipif(not _mojo(), reason="Mojo McKean backend unavailable")
def test_mojo_per_step_within_tolerance() -> None:
    rng = np.random.default_rng(70)
    worst = 0.0
    for _ in range(5000):
        v = float(rng.uniform(-0.3, 1.2))
        w = float(rng.uniform(-0.3, 0.6))
        cur = float(rng.uniform(0.0, 1.0))
        ref, _rs, rv, rw = McKeanNeuron(v=v, w=w)._simulate_python(1, cur)
        got, _gs, gv, gw = McKeanNeuron(v=v, w=w)._simulate_mojo(1, cur)
        worst = max(worst, abs(ref[0] - got[0]), abs(rv - gv), abs(rw - gw))
    assert worst <= _STEP_TOL, f"per-step Mojo gap {worst} exceeds {_STEP_TOL}"


# ───────────────────────────── dispatch + algorithm ───────────────────────────


def test_auto_matches_python() -> None:
    ref, ref_spikes, _rv, _rw = _run("python")
    got, spikes, _vf, _wf = _run("auto")
    np.testing.assert_allclose(got, ref, atol=1e-9, rtol=0.0)
    assert spikes == ref_spikes


def test_invalid_backend_raises() -> None:
    with pytest.raises(ValueError, match="backend must be"):
        McKeanNeuron().simulate(10, 0.0, backend="cuda")


def test_negative_n_steps_raises() -> None:
    with pytest.raises(ValueError, match="n_steps must be non-negative"):
        McKeanNeuron().simulate(-1, 0.0)


def test_simulate_matches_repeated_step() -> None:
    trace_a, spikes_a = McKeanNeuron().simulate(300, 0.5, backend="python")
    manual = []
    spikes_b = 0
    stepper = McKeanNeuron()
    for _ in range(300):
        spikes_b += stepper.step(0.5)
        manual.append(stepper.v)
    np.testing.assert_array_equal(trace_a, np.asarray(manual, dtype=np.float64))
    assert spikes_a == spikes_b


def test_simulate_rejects_invalid_contract() -> None:
    # The runtime contract (a in (0,1), positive scales) is enforced before the loop.
    with pytest.raises((ValueError, FloatingPointError)):
        McKeanNeuron().simulate(10, float("nan"))
