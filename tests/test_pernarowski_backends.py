# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Polyglot parity tests for the Pernarowski 1994 beta-cell burster

"""Cross-backend parity for ``PernarowskiNeuron.simulate``.

Pernarowski's beta-cell model is a three-dimensional slow-fast RK4 burster with an
exact polynomial right-hand side (the cubic is written ``v*v*v`` so it matches the
engine's ``v.powi(3)`` to the last bit; no transcendental functions), so Rust,
Julia and Go reproduce the NumPy reference **bit-for-bit**. Mojo's release build
contracts the RK4 multiply-adds into fused multiply-adds (one rounding instead of
two); the model is a periodic slow-fast burster, so that single-ULP difference
stays bounded over long horizons and the spike counts match.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models import pernarowski
from sc_neurocore.neurons.models.pernarowski import PernarowskiNeuron

_ULP = float(np.spacing(1.0))
_STEP_TOL = 8.0 * _ULP


def _run(backend: str, *, n: int = 4000, current: float = 0.0, **params) -> tuple:
    neuron = PernarowskiNeuron(**params)
    trace, spikes = neuron.simulate(n, current, backend=backend)
    return trace, spikes, neuron.v, neuron.w, neuron.z


def _rust() -> bool:
    return pernarowski._HAS_RUST


def _julia() -> bool:
    return pernarowski._ensure_julia_loaded()


def _go() -> bool:
    return pernarowski._ensure_go_loaded()


def _mojo() -> bool:
    return pernarowski._ensure_mojo_loaded()


_BIT_EXACT = [("rust", _rust), ("julia", _julia), ("go", _go)]
_CURRENTS = [0.0, 0.2, 0.5, 1.0]
_REGIMES = [
    dict(eps1=0.1, eps2=0.001),
    dict(eps1=0.08, eps2=0.0015, beta=0.4),
    dict(eps1=0.12, eps2=0.0008, gamma=0.6),
]


# ───────────────────── bit-exact backends (rust/julia/go) ─────────────────────


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
@pytest.mark.parametrize("current", _CURRENTS)
def test_bit_exact_trace_currents(backend: str, available, current: float) -> None:
    if not available():
        pytest.skip(f"{backend} Pernarowski backend unavailable")
    ref_trace, ref_spikes, rv, rw, rz = _run("python", current=current)
    trace, spikes, vf, wf, zf = _run(backend, current=current)
    np.testing.assert_array_equal(trace, ref_trace)
    assert spikes == ref_spikes
    assert (vf, wf, zf) == (rv, rw, rz)


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
@pytest.mark.parametrize("regime", _REGIMES, ids=["e1", "e2", "e3"])
def test_bit_exact_trace_regimes(backend: str, available, regime: dict) -> None:
    if not available():
        pytest.skip(f"{backend} Pernarowski backend unavailable")
    ref_trace, ref_spikes, rv, rw, rz = _run("python", current=0.3, **regime)
    trace, spikes, vf, wf, zf = _run(backend, current=0.3, **regime)
    np.testing.assert_array_equal(trace, ref_trace)
    assert spikes == ref_spikes
    assert (vf, wf, zf) == (rv, rw, rz)


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
def test_bit_exact_empty_and_single(backend: str, available) -> None:
    if not available():
        pytest.skip(f"{backend} Pernarowski backend unavailable")
    for n in (0, 1, 2):
        ref, rs, rv, rw, rz = _run("python", n=n)
        got, gs, gv, gw, gz = _run(backend, n=n)
        np.testing.assert_array_equal(got, ref)
        assert (gs, gv, gw, gz) == (rs, rv, rw, rz)


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
def test_bit_exact_long_horizon(backend: str, available) -> None:
    if not available():
        pytest.skip(f"{backend} Pernarowski backend unavailable")
    ref, rs, rv, rw, rz = _run("python", n=60_000, current=0.0)
    got, gs, gv, gw, gz = _run(backend, n=60_000, current=0.0)
    np.testing.assert_array_equal(got, ref)
    assert (gs, gv, gw, gz) == (rs, rv, rw, rz)


# ───────────────────────────── Mojo (FMA, non-amplifying) ─────────────────────


@pytest.mark.skipif(not _mojo(), reason="Mojo Pernarowski backend unavailable")
@pytest.mark.parametrize("current", _CURRENTS)
def test_mojo_whole_trace_non_amplifying(current: float) -> None:
    # A periodic slow-fast burster is not chaotic, so the FMA ULP stays bounded:
    # the whole trace stays allclose and the spike count matches exactly.
    ref, ref_spikes, _rv, _rw, _rz = _run("python", n=20_000, current=current)
    got, spikes, _vf, _wf, _zf = _run("mojo", n=20_000, current=current)
    np.testing.assert_allclose(got, ref, atol=1e-9, rtol=0.0)
    assert spikes == ref_spikes


@pytest.mark.skipif(not _mojo(), reason="Mojo Pernarowski backend unavailable")
def test_mojo_per_step_within_tolerance() -> None:
    rng = np.random.default_rng(13)
    worst = 0.0
    for _ in range(5000):
        v = float(rng.uniform(-2.0, 2.0))
        w = float(rng.uniform(-0.5, 1.0))
        z = float(rng.uniform(-0.5, 0.5))
        cur = float(rng.uniform(0.0, 1.0))
        ref, _rs, rv, rw, rz = PernarowskiNeuron(v=v, w=w, z=z)._simulate_python(1, cur)
        got, _gs, gv, gw, gz = PernarowskiNeuron(v=v, w=w, z=z)._simulate_mojo(1, cur)
        worst = max(worst, abs(ref[0] - got[0]), abs(rv - gv), abs(rw - gw), abs(rz - gz))
    assert worst <= _STEP_TOL, f"per-step Mojo gap {worst} exceeds {_STEP_TOL}"


# ───────────────────────────── dispatch + algorithm ───────────────────────────


def test_auto_matches_python() -> None:
    ref, ref_spikes, _rv, _rw, _rz = _run("python")
    got, spikes, _vf, _wf, _zf = _run("auto")
    np.testing.assert_allclose(got, ref, atol=1e-9, rtol=0.0)
    assert spikes == ref_spikes


def test_invalid_backend_raises() -> None:
    with pytest.raises(ValueError, match="backend must be"):
        PernarowskiNeuron().simulate(10, 0.0, backend="cuda")


def test_negative_n_steps_raises() -> None:
    with pytest.raises(ValueError, match="n_steps must be non-negative"):
        PernarowskiNeuron().simulate(-1, 0.0)


def test_simulate_matches_repeated_step() -> None:
    trace_a, spikes_a = PernarowskiNeuron().simulate(400, 0.0, backend="python")
    manual = []
    spikes_b = 0
    stepper = PernarowskiNeuron()
    for _ in range(400):
        spikes_b += stepper.step(0.0)
        manual.append(stepper.v)
    np.testing.assert_array_equal(trace_a, np.asarray(manual, dtype=np.float64))
    assert spikes_a == spikes_b


def test_bursting_produces_spikes() -> None:
    # The default beta-cell regime bursts: many threshold crossings over 20k steps.
    _trace, spikes = PernarowskiNeuron().simulate(20_000, 0.0, backend="python")
    assert spikes > 20
