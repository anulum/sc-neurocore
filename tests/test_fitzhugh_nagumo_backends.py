# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Polyglot parity tests for the FitzHugh-Nagumo RK4 simulator

"""Cross-backend parity for ``FitzHughNagumoNeuron.simulate`` (RK4).

The RK4 right-hand side is exact arithmetic — the cube is written ``v*v*v``
(matching Rust ``v.powi(3)``, Julia ``v^3`` and Go/Mojo ``v*v*v``), with no
transcendental functions — and FitzHugh-Nagumo is a two-dimensional flow, so by
Poincaré-Bendixson it cannot be chaotic. Rust, Julia and Go therefore reproduce
the NumPy reference **bit-for-bit**. Mojo's release build fuses some of the RK4
multiply-adds into FMAs (one rounding instead of two); the per-step gap stays a
couple of ULP and, being non-chaotic, does not amplify — so Mojo is checked on a
tight non-amplifying band with identical spike counts.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models import fitzhugh_nagumo as fhn
from sc_neurocore.neurons.models.fitzhugh_nagumo import FitzHughNagumoNeuron

# Mojo FMA band (measured ~6e-15 over 8k steps); generous and non-amplifying.
_MOJO_ATOL = 1e-11


def _run(backend: str, *, current: float = 0.5, n: int = 8000, **kw) -> tuple:
    neuron = FitzHughNagumoNeuron(**kw)
    trace, spikes = neuron.simulate(n, current, backend=backend)
    return trace, spikes, neuron.v, neuron.w


def _rust() -> bool:
    return fhn._HAS_RUST


def _julia() -> bool:
    return fhn._ensure_julia_loaded()


def _go() -> bool:
    return fhn._ensure_go_loaded()


def _mojo() -> bool:
    return fhn._ensure_mojo_loaded()


_BIT_EXACT = [("rust", _rust), ("julia", _julia), ("go", _go)]
_CURRENTS = [0.0, 0.3, 0.5, 1.0]


# ───────────────────── bit-exact backends (rust/julia/go) ─────────────────────


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
@pytest.mark.parametrize("current", _CURRENTS)
def test_bit_exact_trace(backend: str, available, current: float) -> None:
    if not available():
        pytest.skip(f"{backend} FitzHugh-Nagumo backend unavailable")
    ref_trace, ref_spikes, rv, rw = _run("python", current=current)
    trace, spikes, vf, wf = _run(backend, current=current)
    np.testing.assert_array_equal(trace, ref_trace)
    assert spikes == ref_spikes
    assert vf == rv and wf == rw


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
def test_bit_exact_empty_and_single(backend: str, available) -> None:
    if not available():
        pytest.skip(f"{backend} FitzHugh-Nagumo backend unavailable")
    for n in (0, 1, 2):
        ref, rs, rv, rw = _run("python", n=n)
        got, gs, gv, gw = _run(backend, n=n)
        np.testing.assert_array_equal(got, ref)
        assert (gs, gv, gw) == (rs, rv, rw)


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
def test_bit_exact_limit_cycle_long_run(backend: str, available) -> None:
    # A sustained limit cycle (I = 0.5) over a long horizon stays bit-exact —
    # the exact RHS has no order sensitivity and the 2-D flow cannot diverge.
    if not available():
        pytest.skip(f"{backend} FitzHugh-Nagumo backend unavailable")
    ref, rs, rv, rw = _run("python", current=0.5, n=50000)
    got, gs, gv, gw = _run(backend, current=0.5, n=50000)
    np.testing.assert_array_equal(got, ref)
    assert (gs, gv, gw) == (rs, rv, rw)


# ───────────────────────────── Mojo (FMA, ULP-bounded) ─────────────────────────


@pytest.mark.skipif(not _mojo(), reason="Mojo FitzHugh-Nagumo backend unavailable")
@pytest.mark.parametrize("current", _CURRENTS)
def test_mojo_trace_ulp_bounded_and_exact_spikes(current: float) -> None:
    ref, ref_spikes, _rv, _rw = _run("python", current=current)
    got, spikes, _vf, _wf = _run("mojo", current=current)
    np.testing.assert_allclose(got, ref, atol=_MOJO_ATOL, rtol=0.0)
    assert spikes == ref_spikes


@pytest.mark.skipif(not _mojo(), reason="Mojo FitzHugh-Nagumo backend unavailable")
def test_mojo_band_does_not_amplify() -> None:
    ref, _rs, _rv, _rw = _run("python", current=0.5, n=50000)
    got, _gs, _vf, _wf = _run("mojo", current=0.5, n=50000)
    assert float(np.max(np.abs(got - ref))) < 1e-9


# ───────────────────────────── dispatch + algorithm ───────────────────────────


def test_auto_matches_python_bit_exact() -> None:
    ref, ref_spikes, _rv, _rw = _run("python")
    got, spikes, _vf, _wf = _run("auto")
    np.testing.assert_array_equal(got, ref)
    assert spikes == ref_spikes


def test_invalid_backend_raises() -> None:
    with pytest.raises(ValueError, match="backend must be"):
        FitzHughNagumoNeuron().simulate(10, 0.0, backend="cuda")


def test_negative_n_steps_raises() -> None:
    with pytest.raises(ValueError, match="n_steps must be non-negative"):
        FitzHughNagumoNeuron().simulate(-1, 0.0)


def test_non_finite_current_raises() -> None:
    with pytest.raises(ValueError, match="current must be finite"):
        FitzHughNagumoNeuron().simulate(10, np.inf)


def test_non_rk4_integrator_rejected() -> None:
    # simulate accelerates RK4 only; other integrators must raise, not silently
    # produce RK4 results.
    for integ in ("baseline_euler", "rosenbrock"):
        with pytest.raises(ValueError, match="RK4 integrator only"):
            FitzHughNagumoNeuron(integrator=integ).simulate(10, 0.5)


def test_simulate_matches_repeated_step() -> None:
    trace_a, spikes_a = FitzHughNagumoNeuron().simulate(500, 0.5, backend="python")
    manual = []
    spikes_b = 0
    stepper = FitzHughNagumoNeuron()
    for _ in range(500):
        spikes_b += stepper.step(0.5)
        manual.append(stepper.v)
    np.testing.assert_array_equal(trace_a, np.asarray(manual, dtype=np.float64))
    assert spikes_a == spikes_b


def test_final_state_advances_instance() -> None:
    neuron = FitzHughNagumoNeuron()
    _trace, _spikes = neuron.simulate(500, 0.5, backend="python")
    manual = FitzHughNagumoNeuron()
    for _ in range(500):
        manual.step(0.5)
    assert neuron.v == manual.v and neuron.w == manual.w


def test_tonic_firing_under_drive() -> None:
    _trace, spikes = FitzHughNagumoNeuron().simulate(20000, 0.5, backend="python")
    assert spikes > 5


def test_rest_at_zero_drive_eventually_silent() -> None:
    # Without drive the neuron relaxes to the stable fixed point: no sustained firing.
    _trace, spikes = FitzHughNagumoNeuron().simulate(20000, 0.0, backend="python")
    assert spikes == 0


def test_trace_is_finite() -> None:
    trace, _spikes = FitzHughNagumoNeuron().simulate(50000, 0.5, backend="python")
    assert np.all(np.isfinite(trace))
