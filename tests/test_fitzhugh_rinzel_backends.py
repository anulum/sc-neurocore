# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Polyglot parity tests for the FitzHugh-Rinzel RK4 simulator

"""Cross-backend parity for ``FitzHughRinzelNeuron.simulate`` (RK4).

The RK4 right-hand side is exact arithmetic — the cube is written ``v*v*v``
(matching Rust ``v.powi(3)``, Julia ``v^3`` and Go/Mojo ``v*v*v``), with no
transcendental functions. So Rust, Julia and Go reproduce the NumPy reference
**bit-for-bit**. Mojo's release build fuses some RK4 multiply-adds into FMAs; the
slow ``mu = 1e-4`` recovery keeps the dynamics from being strongly chaotic, so
that per-step ULP stays a small non-amplifying band (measured ~1.5e-12 over
50,000 steps) with identical spike counts.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models import fitzhugh_rinzel as fhr
from sc_neurocore.neurons.models.fitzhugh_rinzel import FitzHughRinzelNeuron

_MOJO_ATOL = 1e-9  # non-amplifying Mojo FMA band (measured ~1.5e-12 at 50k steps)


def _run(backend: str, *, current: float = 0.5, n: int = 8000, **kw) -> tuple:
    neuron = FitzHughRinzelNeuron(**kw)
    trace, spikes = neuron.simulate(n, current, backend=backend)
    return trace, spikes, neuron.v, neuron.w, neuron.y


def _rust() -> bool:
    return fhr._HAS_RUST


def _julia() -> bool:
    return fhr._ensure_julia_loaded()


def _go() -> bool:
    return fhr._ensure_go_loaded()


def _mojo() -> bool:
    return fhr._ensure_mojo_loaded()


_BIT_EXACT = [("rust", _rust), ("julia", _julia), ("go", _go)]
_CURRENTS = [0.0, 0.3, 0.5, 1.0]


# ───────────────────── bit-exact backends (rust/julia/go) ─────────────────────


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
@pytest.mark.parametrize("current", _CURRENTS)
def test_bit_exact_trace(backend: str, available, current: float) -> None:
    if not available():
        pytest.skip(f"{backend} FitzHugh-Rinzel backend unavailable")
    ref_trace, ref_spikes, rv, rw, ry = _run("python", current=current)
    trace, spikes, vf, wf, yf = _run(backend, current=current)
    np.testing.assert_array_equal(trace, ref_trace)
    assert spikes == ref_spikes
    assert (vf, wf, yf) == (rv, rw, ry)


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
def test_bit_exact_empty_and_single(backend: str, available) -> None:
    if not available():
        pytest.skip(f"{backend} FitzHugh-Rinzel backend unavailable")
    for n in (0, 1, 2):
        ref, rs, rv, rw, ry = _run("python", n=n)
        got, gs, gv, gw, gy = _run(backend, n=n)
        np.testing.assert_array_equal(got, ref)
        assert (gs, gv, gw, gy) == (rs, rv, rw, ry)


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
def test_bit_exact_slow_burst_long_run(backend: str, available) -> None:
    # The slow y variable (mu=1e-4) shapes bursts over tens of thousands of
    # steps; the exact RHS keeps the bit-exact backends locked to the reference.
    if not available():
        pytest.skip(f"{backend} FitzHugh-Rinzel backend unavailable")
    ref, rs, rv, rw, ry = _run("python", current=0.5, n=60000)
    got, gs, gv, gw, gy = _run(backend, current=0.5, n=60000)
    np.testing.assert_array_equal(got, ref)
    assert (gs, gv, gw, gy) == (rs, rv, rw, ry)


# ───────────────────────────── Mojo (FMA, ULP-bounded) ─────────────────────────


@pytest.mark.skipif(not _mojo(), reason="Mojo FitzHugh-Rinzel backend unavailable")
@pytest.mark.parametrize("current", _CURRENTS)
def test_mojo_trace_ulp_bounded_and_exact_spikes(current: float) -> None:
    ref, ref_spikes, _rv, _rw, _ry = _run("python", current=current)
    got, spikes, _vf, _wf, _yf = _run("mojo", current=current)
    np.testing.assert_allclose(got, ref, atol=_MOJO_ATOL, rtol=0.0)
    assert spikes == ref_spikes


@pytest.mark.skipif(not _mojo(), reason="Mojo FitzHugh-Rinzel backend unavailable")
def test_mojo_band_does_not_amplify() -> None:
    ref, ref_spikes, _rv, _rw, _ry = _run("python", current=0.5, n=50000)
    got, spikes, _vf, _wf, _yf = _run("mojo", current=0.5, n=50000)
    assert float(np.max(np.abs(got - ref))) < 1e-9
    assert spikes == ref_spikes


# ───────────────────────────── dispatch + algorithm ───────────────────────────


def test_auto_matches_python_bit_exact() -> None:
    ref, ref_spikes, _rv, _rw, _ry = _run("python")
    got, spikes, _vf, _wf, _yf = _run("auto")
    np.testing.assert_array_equal(got, ref)
    assert spikes == ref_spikes


def test_invalid_backend_raises() -> None:
    with pytest.raises(ValueError, match="backend must be"):
        FitzHughRinzelNeuron().simulate(10, 0.0, backend="cuda")


def test_negative_n_steps_raises() -> None:
    with pytest.raises(ValueError, match="n_steps must be non-negative"):
        FitzHughRinzelNeuron().simulate(-1, 0.0)


def test_non_finite_current_raises() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        FitzHughRinzelNeuron().simulate(10, np.inf)


def test_simulate_matches_repeated_step() -> None:
    trace_a, spikes_a = FitzHughRinzelNeuron().simulate(500, 0.5, backend="python")
    manual = []
    spikes_b = 0
    stepper = FitzHughRinzelNeuron()
    for _ in range(500):
        spikes_b += stepper.step(0.5)
        manual.append(stepper.v)
    np.testing.assert_array_equal(trace_a, np.asarray(manual, dtype=np.float64))
    assert spikes_a == spikes_b


def test_final_state_advances_instance() -> None:
    neuron = FitzHughRinzelNeuron()
    _trace, _spikes = neuron.simulate(500, 0.5, backend="python")
    manual = FitzHughRinzelNeuron()
    for _ in range(500):
        manual.step(0.5)
    assert (neuron.v, neuron.w, neuron.y) == (manual.v, manual.w, manual.y)


def test_bursting_under_drive() -> None:
    _trace, spikes = FitzHughRinzelNeuron().simulate(20000, 0.5, backend="python")
    assert spikes > 5


def test_subthreshold_silent() -> None:
    _trace, spikes = FitzHughRinzelNeuron().simulate(20000, 0.0, backend="python")
    assert spikes == 0


def test_trace_is_finite() -> None:
    trace, _spikes = FitzHughRinzelNeuron().simulate(60000, 0.5, backend="python")
    assert np.all(np.isfinite(trace))
