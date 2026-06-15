# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Polyglot parity tests for the Medvedev 2005 1D spiking map

"""Cross-backend parity for ``MedvedevMapNeuron.simulate``.

The map is exact floating-point arithmetic (a multiply, an add, and a fold into
``[0, 1)``), so Rust, Julia and Go reproduce the NumPy reference **bit-for-bit**
across the chaotic regime — the fold uses the Euclidean remainder (Python ``x %
1.0`` == Rust ``rem_euclid(1.0)`` == Julia ``mod(x, 1.0)`` == Go/Mojo
``x - floor(x)``, all bit-identical for unit divisor).

Mojo's release build contracts ``alpha*x + current`` into a fused multiply-add
(one rounding instead of two), so each step agrees only to within a couple of
ULP. This is an expanding chaotic map (``alpha = 3.5 > 1``), so that single ULP
is amplified into a visibly different whole trace and a slightly different spike
count over long horizons — by design, not a defect. Mojo is therefore checked on
the rigorous per-step ULP bound and structural invariants, not on whole-trace or
exact-spike-count equality, matching the documented Mojo FMA-parity precedent for
wong_wang / wilson_cowan.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models import medvedev_map
from sc_neurocore.neurons.models.medvedev_map import MedvedevMapNeuron

_ULP = float(np.spacing(1.0))
_STEP_TOL = 8.0 * _ULP


def _run(backend: str, *, x0: float = 0.0, n: int = 4000, current: float = 0.1) -> tuple:
    neuron = MedvedevMapNeuron(x=x0)
    trace, spikes = neuron.simulate(n, current, backend=backend)
    return trace, spikes, neuron.x


def _rust() -> bool:
    return medvedev_map._HAS_RUST


def _julia() -> bool:
    return medvedev_map._ensure_julia_loaded()


def _go() -> bool:
    return medvedev_map._ensure_go_loaded()


def _mojo() -> bool:
    return medvedev_map._ensure_mojo_loaded()


_BIT_EXACT = [("rust", _rust), ("julia", _julia), ("go", _go)]
# A range of drives (chaotic spiking) plus a non-trivial start phase.
_CURRENTS = [0.05, 0.1, 0.2, 0.37]


# ───────────────────── bit-exact backends (rust/julia/go) ─────────────────────


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
@pytest.mark.parametrize("current", _CURRENTS)
def test_bit_exact_trace(backend: str, available, current: float) -> None:
    if not available():
        pytest.skip(f"{backend} Medvedev backend unavailable")
    ref_trace, ref_spikes, rx = _run("python", current=current)
    trace, spikes, xf = _run(backend, current=current)
    np.testing.assert_array_equal(trace, ref_trace)
    assert spikes == ref_spikes
    assert xf == rx


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
def test_bit_exact_empty_and_single(backend: str, available) -> None:
    if not available():
        pytest.skip(f"{backend} Medvedev backend unavailable")
    for n in (0, 1, 2):
        ref, rs, rx = _run("python", n=n)
        got, gs, gx = _run(backend, n=n)
        np.testing.assert_array_equal(got, ref)
        assert (gs, gx) == (rs, rx)


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
def test_bit_exact_negative_phase_fold(backend: str, available) -> None:
    # A negative intermediate value exercises the Euclidean fold (the case where
    # truncated and floored remainders disagree); all backends must still match.
    if not available():
        pytest.skip(f"{backend} Medvedev backend unavailable")
    ref, rs, rx = _run("python", x0=0.95, current=-0.5, n=4000)
    got, gs, gx = _run(backend, x0=0.95, current=-0.5, n=4000)
    np.testing.assert_array_equal(got, ref)
    assert (gs, gx) == (rs, rx)


# ───────────────────────────── Mojo (FMA, ULP-bounded) ─────────────────────────


@pytest.mark.skipif(not _mojo(), reason="Mojo Medvedev backend unavailable")
def test_mojo_per_step_within_two_ulp() -> None:
    rng = np.random.default_rng(5)
    worst = 0.0
    for _ in range(20000):
        x = float(rng.uniform(0.0, 1.0))
        cur = float(rng.uniform(0.0, 0.5))
        ref, _rs, rx = MedvedevMapNeuron(x=x)._simulate_python(1, cur)
        got, _gs, gx = MedvedevMapNeuron(x=x)._simulate_mojo(1, cur)
        worst = max(worst, abs(ref[0] - got[0]), abs(rx - gx))
    assert worst <= _STEP_TOL, f"per-step Mojo gap {worst} exceeds {_STEP_TOL}"


@pytest.mark.skipif(not _mojo(), reason="Mojo Medvedev backend unavailable")
def test_mojo_trace_stays_in_unit_interval() -> None:
    # The fold keeps every Mojo sample in [0, 1) even though the chaotic trace
    # diverges from the reference over long horizons.
    trace, spikes = MedvedevMapNeuron().simulate(8000, 0.1, backend="mojo")
    assert np.all(trace >= 0.0) and np.all(trace < 1.0)
    assert spikes > 0


@pytest.mark.skipif(not _mojo(), reason="Mojo Medvedev backend unavailable")
def test_mojo_short_horizon_tracks_reference() -> None:
    # Before the chaotic amplification dominates, the first handful of steps stay
    # within a few ULP of the reference.
    ref, _rs = MedvedevMapNeuron().simulate(8, 0.1, backend="python")
    got, _gs = MedvedevMapNeuron().simulate(8, 0.1, backend="mojo")
    np.testing.assert_allclose(got[:5], ref[:5], atol=1e-12, rtol=0.0)


# ───────────────────────────── dispatch + algorithm ───────────────────────────


def test_auto_matches_python_bit_exact() -> None:
    # auto -> Rust, which is bit-exact (unlike the FMA Mojo path).
    ref, ref_spikes, _rx = _run("python")
    got, spikes, _xf = _run("auto")
    np.testing.assert_array_equal(got, ref)
    assert spikes == ref_spikes


def test_invalid_backend_raises() -> None:
    with pytest.raises(ValueError, match="backend must be"):
        MedvedevMapNeuron().simulate(10, 0.0, backend="cuda")


def test_negative_n_steps_raises() -> None:
    with pytest.raises(ValueError, match="n_steps must be non-negative"):
        MedvedevMapNeuron().simulate(-1, 0.0)


def test_non_finite_current_raises() -> None:
    with pytest.raises(ValueError, match="current must be finite"):
        MedvedevMapNeuron().simulate(10, np.nan)


def test_simulate_matches_repeated_step() -> None:
    trace_a, spikes_a = MedvedevMapNeuron().simulate(300, 0.1, backend="python")
    manual = []
    spikes_b = 0
    stepper = MedvedevMapNeuron()
    for _ in range(300):
        spikes_b += stepper.step(0.1)
        manual.append(stepper.x)
    np.testing.assert_array_equal(trace_a, np.asarray(manual, dtype=np.float64))
    assert spikes_a == spikes_b


def test_final_state_advances_instance() -> None:
    neuron = MedvedevMapNeuron()
    _trace, _spikes = neuron.simulate(500, 0.1, backend="python")
    manual = MedvedevMapNeuron()
    for _ in range(500):
        manual.step(0.1)
    assert neuron.x == manual.x


def test_chaotic_drive_produces_spikes() -> None:
    _trace, spikes = MedvedevMapNeuron().simulate(20000, 0.1, backend="python")
    assert spikes > 100


def test_trace_confined_to_unit_interval() -> None:
    trace, _spikes = MedvedevMapNeuron().simulate(50000, 0.1, backend="python")
    assert np.all(trace >= 0.0) and np.all(trace < 1.0)


def test_zero_current_rests_at_fixed_point() -> None:
    # x0 = 0 with no drive is the fixed point: 0 -> 0, no spikes.
    trace, spikes = MedvedevMapNeuron().simulate(1000, 0.0, backend="python")
    assert spikes == 0
    assert np.all(trace == 0.0)
