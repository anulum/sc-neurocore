# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Polyglot parity tests for the Hindmarsh-Rose RK4 simulator

"""Cross-backend parity for ``HindmarshRoseNeuron.simulate`` (RK4).

The RK4 right-hand side is exact arithmetic — the square and cube are written
``x*x`` and ``(x*x)*x`` (matching Rust ``x.powi(2)``/``x.powi(3)``, Julia
``x^2``/``x^3`` and Go/Mojo ``x*x``), with no transcendental functions. So even
though Hindmarsh-Rose is a three-dimensional **chaotic** burster, Rust, Julia and
Go reproduce the NumPy reference **bit-for-bit** at every horizon — exactness is
independent of the dynamics. Mojo's release build fuses some RK4 multiply-adds
into FMAs; that per-step ULP (<=8 ULP) is amplified by the chaotic flow into a
divergent whole trace, so Mojo is checked on the per-step bound and structural
invariants rather than whole-trace equality.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models import hindmarsh_rose as hr
from sc_neurocore.neurons.models.hindmarsh_rose import HindmarshRoseNeuron

_STEP_TOL = 8e-15  # per-step Mojo FMA bound (measured worst ~1.8e-15 over x/y/z)


def _run(backend: str, *, current: float = 3.0, n: int = 8000, **kw) -> tuple:
    neuron = HindmarshRoseNeuron(**kw)
    trace, spikes = neuron.simulate(n, current, backend=backend)
    return trace, spikes, neuron.x, neuron.y, neuron.z


def _rust() -> bool:
    return hr._HAS_RUST


def _julia() -> bool:
    return hr._ensure_julia_loaded()


def _go() -> bool:
    return hr._ensure_go_loaded()


def _mojo() -> bool:
    return hr._ensure_mojo_loaded()


_BIT_EXACT = [("rust", _rust), ("julia", _julia), ("go", _go)]
_CURRENTS = [0.0, 1.0, 2.0, 3.2]


# ───────────────────── bit-exact backends (rust/julia/go) ─────────────────────


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
@pytest.mark.parametrize("current", _CURRENTS)
def test_bit_exact_trace(backend: str, available, current: float) -> None:
    if not available():
        pytest.skip(f"{backend} Hindmarsh-Rose backend unavailable")
    ref_trace, ref_spikes, rx, ry, rz = _run("python", current=current)
    trace, spikes, xf, yf, zf = _run(backend, current=current)
    np.testing.assert_array_equal(trace, ref_trace)
    assert spikes == ref_spikes
    assert (xf, yf, zf) == (rx, ry, rz)


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
def test_bit_exact_empty_and_single(backend: str, available) -> None:
    if not available():
        pytest.skip(f"{backend} Hindmarsh-Rose backend unavailable")
    for n in (0, 1, 2):
        ref, rs, rx, ry, rz = _run("python", n=n)
        got, gs, gx, gy, gz = _run(backend, n=n)
        np.testing.assert_array_equal(got, ref)
        assert (gs, gx, gy, gz) == (rs, rx, ry, rz)


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
def test_bit_exact_chaotic_long_run(backend: str, available) -> None:
    # Exact arithmetic stays bit-exact even across a long chaotic bursting run,
    # where a transcendental or FMA backend would have diverged completely.
    if not available():
        pytest.skip(f"{backend} Hindmarsh-Rose backend unavailable")
    ref, rs, rx, ry, rz = _run("python", current=3.0, n=60000)
    got, gs, gx, gy, gz = _run(backend, current=3.0, n=60000)
    np.testing.assert_array_equal(got, ref)
    assert (gs, gx, gy, gz) == (rs, rx, ry, rz)


# ───────────────────────────── Mojo (FMA, ULP-bounded) ─────────────────────────


@pytest.mark.skipif(not _mojo(), reason="Mojo Hindmarsh-Rose backend unavailable")
def test_mojo_per_step_within_tolerance() -> None:
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(5000):
        x = float(rng.uniform(-2.0, 2.0))
        y = float(rng.uniform(-12.0, 2.0))
        z = float(rng.uniform(0.0, 4.0))
        cur = float(rng.uniform(0.0, 4.0))
        ref, _rs, rx, ry, rz = HindmarshRoseNeuron(x=x, y=y, z=z)._simulate_python(1, cur)
        got, _gs, gx, gy, gz = HindmarshRoseNeuron(x=x, y=y, z=z)._simulate_mojo(1, cur)
        worst = max(worst, abs(ref[0] - got[0]), abs(rx - gx), abs(ry - gy), abs(rz - gz))
    assert worst <= _STEP_TOL, f"per-step Mojo gap {worst} exceeds {_STEP_TOL}"


@pytest.mark.skipif(not _mojo(), reason="Mojo Hindmarsh-Rose backend unavailable")
def test_mojo_trace_finite_and_short_horizon_tracks() -> None:
    # Over a short horizon (before chaos amplifies the FMA ULP) Mojo tracks the
    # reference closely; the full trace stays finite.
    ref, _rs, _rx, _ry, _rz = _run("python", current=3.0, n=8000)
    got, spikes, _gx, _gy, _gz = _run("mojo", current=3.0, n=8000)
    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(got[:200], ref[:200], atol=1e-12, rtol=0.0)
    assert spikes == _rs


# ───────────────────────────── dispatch + algorithm ───────────────────────────


def test_auto_matches_python_bit_exact() -> None:
    ref, ref_spikes, _rx, _ry, _rz = _run("python")
    got, spikes, _xf, _yf, _zf = _run("auto")
    np.testing.assert_array_equal(got, ref)
    assert spikes == ref_spikes


def test_invalid_backend_raises() -> None:
    with pytest.raises(ValueError, match="backend must be"):
        HindmarshRoseNeuron().simulate(10, 0.0, backend="cuda")


def test_negative_n_steps_raises() -> None:
    with pytest.raises(ValueError, match="n_steps must be non-negative"):
        HindmarshRoseNeuron().simulate(-1, 0.0)


def test_non_finite_current_raises() -> None:
    with pytest.raises(ValueError, match="current must be finite"):
        HindmarshRoseNeuron().simulate(10, np.nan)


def test_non_rk4_integrator_rejected() -> None:
    with pytest.raises(ValueError, match="RK4 integrator only"):
        HindmarshRoseNeuron(integrator="euler").simulate(10, 3.0)


def test_simulate_matches_repeated_step() -> None:
    trace_a, spikes_a = HindmarshRoseNeuron().simulate(500, 3.0, backend="python")
    manual = []
    spikes_b = 0
    stepper = HindmarshRoseNeuron()
    for _ in range(500):
        spikes_b += stepper.step(3.0)
        manual.append(stepper.x)
    np.testing.assert_array_equal(trace_a, np.asarray(manual, dtype=np.float64))
    assert spikes_a == spikes_b


def test_final_state_advances_instance() -> None:
    neuron = HindmarshRoseNeuron()
    _trace, _spikes = neuron.simulate(500, 3.0, backend="python")
    manual = HindmarshRoseNeuron()
    for _ in range(500):
        manual.step(3.0)
    assert (neuron.x, neuron.y, neuron.z) == (manual.x, manual.y, manual.z)


def test_bursting_under_drive() -> None:
    _trace, spikes = HindmarshRoseNeuron().simulate(20000, 3.0, backend="python")
    assert spikes > 10


def test_subthreshold_silent() -> None:
    _trace, spikes = HindmarshRoseNeuron().simulate(20000, 0.0, backend="python")
    assert spikes == 0


def test_trace_is_finite() -> None:
    trace, _spikes = HindmarshRoseNeuron().simulate(60000, 3.0, backend="python")
    assert np.all(np.isfinite(trace))
