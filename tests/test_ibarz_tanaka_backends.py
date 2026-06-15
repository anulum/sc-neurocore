# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Polyglot parity tests for the Ibarz-Tanaka piecewise-linear map

"""Cross-backend parity for ``IbarzTanakaMapNeuron.simulate``.

The map is exact floating-point arithmetic (one division, additions and
multiplications, no transcendental functions), so Rust, Julia and Go reproduce
the NumPy reference **bit-for-bit** across the bursting and spiking regimes.
Mojo's release build can contract ``alpha + beta*x`` and ``y - mu*(x+1) +
mu*sigma`` into fused multiply-adds (one rounding instead of two); each step
therefore agrees to within a couple of ULP. The explicit reset to ``x_reset``
on every spike periodically resynchronises the trajectory, so the whole-trace
gap stays at the per-step ULP level — Mojo is checked to a documented ULP bound
rather than bit-for-bit, matching the Mojo FMA-parity precedent for
wong_wang / wilson_cowan.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models import ibarz_tanaka_map
from sc_neurocore.neurons.models.ibarz_tanaka_map import IbarzTanakaMapNeuron

_ULP = float(np.spacing(1.0))
_STEP_TOL = 8.0 * _ULP


def _run(backend: str, *, sigma: float = -1.6, n: int = 4000, current: float = 0.1) -> tuple:
    neuron = IbarzTanakaMapNeuron(sigma=sigma)
    trace, spikes = neuron.simulate(n, current, backend=backend)
    return trace, spikes, neuron.x, neuron.y


def _rust() -> bool:
    return ibarz_tanaka_map._HAS_RUST


def _julia() -> bool:
    return ibarz_tanaka_map._ensure_julia_loaded()


def _go() -> bool:
    return ibarz_tanaka_map._ensure_go_loaded()


def _mojo() -> bool:
    return ibarz_tanaka_map._ensure_mojo_loaded()


_BIT_EXACT = [("rust", _rust), ("julia", _julia), ("go", _go)]
# Silent, near-threshold, bursting and strongly-driven regimes.
_REGIMES = [-1.6, -0.8, 0.0, 1.0]


# ───────────────────── bit-exact backends (rust/julia/go) ─────────────────────


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
@pytest.mark.parametrize("sigma", _REGIMES)
def test_bit_exact_trace(backend: str, available, sigma: float) -> None:
    if not available():
        pytest.skip(f"{backend} Ibarz-Tanaka backend unavailable")
    ref_trace, ref_spikes, rx, ry = _run("python", sigma=sigma)
    trace, spikes, xf, yf = _run(backend, sigma=sigma)
    np.testing.assert_array_equal(trace, ref_trace)
    assert spikes == ref_spikes
    assert xf == rx and yf == ry


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
def test_bit_exact_empty_and_single(backend: str, available) -> None:
    if not available():
        pytest.skip(f"{backend} Ibarz-Tanaka backend unavailable")
    for n in (0, 1, 2):
        ref, rs, rx, ry = _run("python", n=n)
        got, gs, gx, gy = _run(backend, n=n)
        np.testing.assert_array_equal(got, ref)
        assert (gs, gx, gy) == (rs, rx, ry)


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
def test_bit_exact_strong_drive_resets(backend: str, available) -> None:
    # High constant drive forces frequent threshold crossings + resets.
    if not available():
        pytest.skip(f"{backend} Ibarz-Tanaka backend unavailable")
    ref, rs, rx, ry = _run("python", current=5.0, n=8000)
    got, gs, gx, gy = _run(backend, current=5.0, n=8000)
    np.testing.assert_array_equal(got, ref)
    assert (gs, gx, gy) == (rs, rx, ry)


# ───────────────────────────── Mojo (FMA, ULP-bounded) ─────────────────────────


@pytest.mark.skipif(not _mojo(), reason="Mojo Ibarz-Tanaka backend unavailable")
@pytest.mark.parametrize("sigma", _REGIMES)
def test_mojo_trace_ulp_bounded(sigma: float) -> None:
    ref, _ref_spikes, _rx, _ry = _run("python", sigma=sigma, n=2000)
    got, _spikes, _xf, _yf = _run("mojo", sigma=sigma, n=2000)
    np.testing.assert_allclose(got, ref, atol=1e-9, rtol=0.0)


@pytest.mark.skipif(not _mojo(), reason="Mojo Ibarz-Tanaka backend unavailable")
def test_mojo_per_step_within_tolerance() -> None:
    rng = np.random.default_rng(13)
    worst = 0.0
    for _ in range(5000):
        x = float(rng.uniform(-2.0, 2.5))
        y = float(rng.uniform(-3.0, -2.0))
        cur = float(rng.uniform(0.0, 2.0))
        ref, _rs, rx, ry = IbarzTanakaMapNeuron(x=x, y=y)._simulate_python(1, cur)
        got, _gs, gx, gy = IbarzTanakaMapNeuron(x=x, y=y)._simulate_mojo(1, cur)
        worst = max(worst, abs(ref[0] - got[0]), abs(rx - gx), abs(ry - gy))
    assert worst <= _STEP_TOL, f"per-step Mojo gap {worst} exceeds {_STEP_TOL}"


@pytest.mark.skipif(not _mojo(), reason="Mojo Ibarz-Tanaka backend unavailable")
def test_mojo_spike_count_matches() -> None:
    # current=3.0 drives the spiking branch (defaults are silent below ~2.0).
    _ref, ref_spikes, _rx, _ry = _run("python", current=3.0, n=8000)
    _got, spikes, _xf, _yf = _run("mojo", current=3.0, n=8000)
    assert ref_spikes > 0
    assert spikes == ref_spikes


# ───────────────────────────── dispatch + algorithm ───────────────────────────


def test_auto_matches_python() -> None:
    ref, ref_spikes, _rx, _ry = _run("python")
    got, spikes, _xf, _yf = _run("auto")
    np.testing.assert_allclose(got, ref, atol=1e-9, rtol=0.0)
    assert spikes == ref_spikes


def test_invalid_backend_raises() -> None:
    with pytest.raises(ValueError, match="backend must be"):
        IbarzTanakaMapNeuron().simulate(10, 0.0, backend="cuda")


def test_negative_n_steps_raises() -> None:
    with pytest.raises(ValueError, match="n_steps must be non-negative"):
        IbarzTanakaMapNeuron().simulate(-1, 0.0)


def test_non_finite_current_raises() -> None:
    with pytest.raises(ValueError, match="current must be finite"):
        IbarzTanakaMapNeuron().simulate(10, np.inf)


def test_simulate_matches_repeated_step() -> None:
    # The N-step path must equal calling step() N times (same state evolution,
    # same reset-on-spike trace, same spike count).
    trace_a, spikes_a = IbarzTanakaMapNeuron().simulate(300, 1.0, backend="python")
    manual = []
    spikes_b = 0
    stepper = IbarzTanakaMapNeuron()
    for _ in range(300):
        spikes_b += stepper.step(1.0)
        manual.append(stepper.x)
    np.testing.assert_array_equal(trace_a, np.asarray(manual, dtype=np.float64))
    assert spikes_a == spikes_b


def test_final_state_advances_instance() -> None:
    neuron = IbarzTanakaMapNeuron()
    _trace, _spikes = neuron.simulate(500, 1.0, backend="python")
    manual = IbarzTanakaMapNeuron()
    for _ in range(500):
        manual.step(1.0)
    assert neuron.x == manual.x and neuron.y == manual.y


def test_strong_drive_produces_spikes() -> None:
    # Defaults are silent below ~2.0; current=3.0 produces sustained bursting.
    _trace, spikes = IbarzTanakaMapNeuron().simulate(20000, 3.0, backend="python")
    assert spikes > 10


def test_trace_resets_to_reset_value_on_spike() -> None:
    # Every spiking step writes x_reset into the trace.
    neuron = IbarzTanakaMapNeuron(x_reset=-1.0)
    trace, spikes = neuron.simulate(20000, 3.0, backend="python")
    assert spikes > 0
    assert np.count_nonzero(np.isclose(trace, -1.0)) >= spikes


def test_long_run_is_finite() -> None:
    trace, _spikes = IbarzTanakaMapNeuron().simulate(100_000, 1.0, backend="python")
    assert np.all(np.isfinite(trace))
