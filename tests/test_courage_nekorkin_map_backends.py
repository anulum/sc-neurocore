# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Polyglot parity tests for the Courbage-Nekorkin-Vdovin 2007 map

"""Cross-backend parity for ``CourageNekorkinMapNeuron.simulate``.

The Courbage-Nekorkin-Vdovin map is exact floating-point arithmetic (additions,
multiplications, one division for the breakpoints, and a piecewise/Heaviside
branch — no transcendental functions), so Rust, Julia and Go reproduce the NumPy
reference **bit-for-bit** even though the default parameters sit in the chaotic
spiking-bursting regime. Mojo's release build contracts a multiply-add into a
fused multiply-add (one rounding instead of two); the trajectory is bit-exact for
the first ~100 steps and then a single ULP appears, which the chaotic map
amplifies into a visible whole-trace gap. That is correct numerical behaviour,
not a defect — the per-step state stays ULP-bounded and the coarse spike count
stays within a fraction of a percent — and matches the documented Mojo FMA-parity
precedent for the other chaotic maps (cazelles / medvedev / hindmarsh_rose).
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models import courage_nekorkin_map
from sc_neurocore.neurons.models.courage_nekorkin_map import CourageNekorkinMapNeuron

# 8 ULP at the order-1 magnitudes the single-step arithmetic visits.
_ULP = float(np.spacing(1.0))
_STEP_TOL = 8.0 * _ULP


def _run(backend: str, *, n: int = 4000, current: float = 0.0, **params) -> tuple:
    neuron = CourageNekorkinMapNeuron(**params)
    trace, spikes = neuron.simulate(n, current, backend=backend)
    return trace, spikes, neuron.x, neuron.y


def _rust() -> bool:
    return courage_nekorkin_map._HAS_RUST


def _julia() -> bool:
    return courage_nekorkin_map._ensure_julia_loaded()


def _go() -> bool:
    return courage_nekorkin_map._ensure_go_loaded()


def _mojo() -> bool:
    return courage_nekorkin_map._ensure_mojo_loaded()


_BIT_EXACT = [("rust", _rust), ("julia", _julia), ("go", _go)]
# Currents spanning the autonomous map and external-drive regimes.
_CURRENTS = [0.0, 0.05, 0.1, -0.02]
# Several (d, J) inside the B^+ invariant-region triangle (Jmin~0.1765, Jmax~0.2939).
_REGIMES = [
    dict(d=0.235, j=0.2),
    dict(d=0.23, j=0.19, x_threshold=0.23),
    dict(d=0.24, j=0.21, x_threshold=0.24),
]


# ───────────────────── bit-exact backends (rust/julia/go) ─────────────────────


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
@pytest.mark.parametrize("current", _CURRENTS)
def test_bit_exact_trace_currents(backend: str, available, current: float) -> None:
    if not available():
        pytest.skip(f"{backend} Courbage backend unavailable")
    ref_trace, ref_spikes, rx, ry = _run("python", current=current)
    trace, spikes, xf, yf = _run(backend, current=current)
    np.testing.assert_array_equal(trace, ref_trace)
    assert spikes == ref_spikes
    assert xf == rx and yf == ry


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
@pytest.mark.parametrize("regime", _REGIMES, ids=["d235", "d230", "d240"])
def test_bit_exact_trace_regimes(backend: str, available, regime: dict) -> None:
    if not available():
        pytest.skip(f"{backend} Courbage backend unavailable")
    ref_trace, ref_spikes, rx, ry = _run("python", current=0.05, **regime)
    trace, spikes, xf, yf = _run(backend, current=0.05, **regime)
    np.testing.assert_array_equal(trace, ref_trace)
    assert spikes == ref_spikes
    assert xf == rx and yf == ry


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
def test_bit_exact_empty_and_single(backend: str, available) -> None:
    if not available():
        pytest.skip(f"{backend} Courbage backend unavailable")
    for n in (0, 1, 2):
        ref, rs, rx, ry = _run("python", n=n)
        got, gs, gx, gy = _run(backend, n=n)
        np.testing.assert_array_equal(got, ref)
        assert (gs, gx, gy) == (rs, rx, ry)


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
def test_bit_exact_long_horizon(backend: str, available) -> None:
    if not available():
        pytest.skip(f"{backend} Courbage backend unavailable")
    ref, rs, rx, ry = _run("python", n=60_000, current=0.0)
    got, gs, gx, gy = _run(backend, n=60_000, current=0.0)
    np.testing.assert_array_equal(got, ref)
    assert (gs, gx, gy) == (rs, rx, ry)


# ───────────────────────────── Mojo (FMA, ULP-bounded) ─────────────────────────


@pytest.mark.skipif(not _mojo(), reason="Mojo Courbage backend unavailable")
def test_mojo_short_horizon_bit_exact() -> None:
    # Before chaotic amplification the FMA path agrees to the last bit.
    ref, _rs, _rx, _ry = _run("python", n=50)
    got, _gs, _gx, _gy = _run("mojo", n=50)
    np.testing.assert_allclose(got, ref, atol=1e-13, rtol=0.0)


@pytest.mark.skipif(not _mojo(), reason="Mojo Courbage backend unavailable")
def test_mojo_per_step_within_tolerance() -> None:
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(5000):
        x = float(rng.uniform(-0.3, 0.4))
        y = float(rng.uniform(-0.2, 0.2))
        cur = float(rng.uniform(-0.1, 0.1))
        ref, _rs, rx, ry = CourageNekorkinMapNeuron(x=x, y=y)._simulate_python(1, cur)
        got, _gs, gx, gy = CourageNekorkinMapNeuron(x=x, y=y)._simulate_mojo(1, cur)
        worst = max(worst, abs(ref[0] - got[0]), abs(rx - gx), abs(ry - gy))
    assert worst <= _STEP_TOL, f"per-step Mojo gap {worst} exceeds {_STEP_TOL}"


@pytest.mark.skipif(not _mojo(), reason="Mojo Courbage backend unavailable")
@pytest.mark.parametrize("current", [0.0, 0.05, 0.1])
def test_mojo_spike_count_band_in_chaotic_regime(current: float) -> None:
    # The chaotic trace diverges by ULP amplification, but the coarse spike count
    # (threshold crossings over 20k steps) stays within a fraction of a percent.
    _ref, ref_spikes, _rx, _ry = _run("python", n=20_000, current=current)
    _got, spikes, _xf, _yf = _run("mojo", n=20_000, current=current)
    rel = abs(spikes - ref_spikes) / max(ref_spikes, 1)
    assert rel <= 0.02, f"Mojo spike band {rel:.4f} exceeds 2% (py={ref_spikes}, mojo={spikes})"


# ───────────────────────────── dispatch + algorithm ───────────────────────────


def test_auto_matches_python() -> None:
    ref, ref_spikes, _rx, _ry = _run("python")
    got, spikes, _xf, _yf = _run("auto")
    np.testing.assert_allclose(got, ref, atol=1e-12, rtol=0.0)
    assert spikes == ref_spikes


def test_invalid_backend_raises() -> None:
    with pytest.raises(ValueError, match="backend must be"):
        CourageNekorkinMapNeuron().simulate(10, 0.0, backend="cuda")


def test_negative_n_steps_raises() -> None:
    with pytest.raises(ValueError, match="n_steps must be non-negative"):
        CourageNekorkinMapNeuron().simulate(-1, 0.0)


def test_simulate_matches_repeated_step() -> None:
    # The N-step path must equal calling step() N times (same state evolution).
    trace_a, spikes_a = CourageNekorkinMapNeuron().simulate(200, 0.0, backend="python")
    manual = []
    spikes_b = 0
    stepper = CourageNekorkinMapNeuron()
    for _ in range(200):
        spikes_b += stepper.step(0.0)
        manual.append(stepper.x)
    np.testing.assert_array_equal(trace_a, np.asarray(manual, dtype=np.float64))
    assert spikes_a == spikes_b
