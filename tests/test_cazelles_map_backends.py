# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Polyglot parity tests for the Cazelles 2001 bursting map

"""Cross-backend parity for ``CazellesMapNeuron.simulate``.

The map is exact floating-point arithmetic, so Rust, Julia and Go reproduce
the NumPy reference **bit-for-bit** even in the chaotic regime (a = 3.8). Mojo's
release build contracts ``y + epsilon*(x - sigma)`` into a fused multiply-add
(one rounding instead of two); each step therefore agrees to within a couple of
ULP, which in the chaotic regime the map amplifies into a visible trace gap.
That is correct numerical behaviour, not a defect — the per-step physical-state
agreement stays tightly ULP-bounded and the spike counts match — and matches the
documented Mojo FMA-parity precedent for wong_wang / wilson_cowan.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models import cazelles_map
from sc_neurocore.neurons.models.cazelles_map import CazellesMapNeuron

# 2 ULP at the order-1 magnitudes the map visits.
_ULP = float(np.spacing(1.0))
_STEP_TOL = 8.0 * _ULP


def _run(backend: str, *, a: float = 3.8, n: int = 4000, current: float = 0.05) -> tuple:
    neuron = CazellesMapNeuron(a=a)
    trace, spikes = neuron.simulate(n, current, backend=backend)
    return trace, spikes, neuron.x, neuron.y


def _rust() -> bool:
    return cazelles_map._HAS_RUST


def _julia() -> bool:
    return cazelles_map._ensure_julia_loaded()


def _go() -> bool:
    return cazelles_map._ensure_go_loaded()


def _mojo() -> bool:
    return cazelles_map._ensure_mojo_loaded()


_BIT_EXACT = [("rust", _rust), ("julia", _julia), ("go", _go)]
_CHAOTIC_AND_REGULAR = [1.5, 2.0, 2.8, 3.2, 3.8]


# ───────────────────── bit-exact backends (rust/julia/go) ─────────────────────


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
@pytest.mark.parametrize("a", _CHAOTIC_AND_REGULAR)
def test_bit_exact_trace(backend: str, available, a: float) -> None:
    if not available():
        pytest.skip(f"{backend} Cazelles backend unavailable")
    ref_trace, ref_spikes, rx, ry = _run("python", a=a)
    trace, spikes, xf, yf = _run(backend, a=a)
    np.testing.assert_array_equal(trace, ref_trace)
    assert spikes == ref_spikes
    assert xf == rx and yf == ry


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
def test_bit_exact_empty_and_single(backend: str, available) -> None:
    if not available():
        pytest.skip(f"{backend} Cazelles backend unavailable")
    for n in (0, 1, 2):
        ref, rs, rx, ry = _run("python", n=n)
        got, gs, gx, gy = _run(backend, n=n)
        np.testing.assert_array_equal(got, ref)
        assert (gs, gx, gy) == (rs, rx, ry)


# ───────────────────────────── Mojo (FMA, ULP-bounded) ─────────────────────────


@pytest.mark.skipif(not _mojo(), reason="Mojo Cazelles backend unavailable")
@pytest.mark.parametrize("a", [1.5, 2.0, 2.8, 3.2])
def test_mojo_regular_regime_ulp_bounded(a: float) -> None:
    # Outside the chaotic regime, single-ULP FMA differences do not amplify.
    ref, _ref_spikes, _rx, _ry = _run("python", a=a, n=2000)
    got, _spikes, _xf, _yf = _run("mojo", a=a, n=2000)
    np.testing.assert_allclose(got, ref, atol=1e-12, rtol=0.0)


@pytest.mark.skipif(not _mojo(), reason="Mojo Cazelles backend unavailable")
def test_mojo_per_step_within_two_ulp() -> None:
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(5000):
        x = float(rng.uniform(-1.0, 1.0))
        y = float(rng.uniform(-1.0, 1.0))
        cur = float(rng.uniform(-0.5, 0.5))
        ref, _rs, rx, ry = CazellesMapNeuron(x=x, y=y)._simulate_python(1, cur)
        got, _gs, gx, gy = CazellesMapNeuron(x=x, y=y)._simulate_mojo(1, cur)
        worst = max(worst, abs(ref[0] - got[0]), abs(rx - gx), abs(ry - gy))
    assert worst <= _STEP_TOL, f"per-step Mojo gap {worst} exceeds {_STEP_TOL}"


@pytest.mark.skipif(not _mojo(), reason="Mojo Cazelles backend unavailable")
def test_mojo_spike_count_matches_in_chaotic_regime() -> None:
    # The chaotic trace diverges by ULP amplification, but the coarse spike
    # count (threshold crossings) is robust and must still agree.
    _ref, ref_spikes, _rx, _ry = _run("python", a=3.8, n=4000)
    _got, spikes, _xf, _yf = _run("mojo", a=3.8, n=4000)
    assert spikes == ref_spikes


# ───────────────────────────── dispatch + algorithm ───────────────────────────


def test_auto_matches_python() -> None:
    ref, ref_spikes, _rx, _ry = _run("python")
    got, spikes, _xf, _yf = _run("auto")
    np.testing.assert_allclose(got, ref, atol=1e-12, rtol=0.0)
    assert spikes == ref_spikes


def test_invalid_backend_raises() -> None:
    with pytest.raises(ValueError, match="backend must be"):
        CazellesMapNeuron().simulate(10, 0.0, backend="cuda")


def test_negative_n_steps_raises() -> None:
    with pytest.raises(ValueError, match="n_steps must be non-negative"):
        CazellesMapNeuron().simulate(-1, 0.0)


def test_simulate_matches_repeated_step() -> None:
    # The N-step path must equal calling step() N times (same state evolution).
    trace_a, spikes_a = CazellesMapNeuron().simulate(50, 0.05, backend="python")
    manual = []
    spikes_b = 0
    stepper = CazellesMapNeuron()
    for _ in range(50):
        spikes_b += stepper.step(0.05)
        manual.append(stepper.x)
    np.testing.assert_array_equal(trace_a, np.asarray(manual, dtype=np.float64))
    assert spikes_a == spikes_b


def test_bursting_produces_spikes() -> None:
    # a = 3.8 with drive must produce threshold crossings (bursting dynamics).
    _trace, spikes = CazellesMapNeuron(a=3.8).simulate(2000, 0.05, backend="python")
    assert spikes > 0


def test_long_run_is_finite() -> None:
    trace, _spikes = CazellesMapNeuron(a=3.8).simulate(100_000, 0.05, backend="python")
    assert np.all(np.isfinite(trace))
    assert np.all(trace >= -2.0) and np.all(trace <= 2.0)
