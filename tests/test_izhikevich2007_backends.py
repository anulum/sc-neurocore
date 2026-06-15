# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Polyglot parity tests for the Izhikevich 2007 RK4 simulator

"""Cross-backend parity for ``Izhikevich2007Neuron.simulate`` (RK4).

The NeuroML right-hand side ``k (v-vr)(v-vt)/C`` is exact arithmetic — products,
a sum and a division, with no transcendental functions — so Rust, Julia and Go
reproduce the NumPy reference **bit-for-bit**. Mojo's release build fuses some
RK4 multiply-adds into FMAs; the hard ``v >= vpeak -> v = c`` reset re-anchors
the trajectory on every spike, so the per-step ULP does not amplify and the
spike counts always match. The residual whole-trace gap depends on the firing
rate: at strong drive (frequent resets) it is ~5e-12, while in a sparse-firing
regime (long inter-spike intervals, few resets) the FMA ULP drifts to ~4e-8 over
8,000 steps. Mojo is therefore validated on a generous absolute band plus exact
spike counts, not bit-for-bit.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models import izhikevich2007 as izh
from sc_neurocore.neurons.models.izhikevich2007 import Izhikevich2007Neuron

# Mojo FMA band: ~5e-12 at strong drive, up to ~4e-8 when firing is sparse.
_MOJO_ATOL = 1e-6


def _run(backend: str, *, current: float = 300.0, n: int = 8000, **kw) -> tuple:
    neuron = Izhikevich2007Neuron(**kw)
    trace, spikes = neuron.simulate(n, current, backend=backend)
    return trace, spikes, neuron.v, neuron.u


def _rust() -> bool:
    return izh._HAS_RUST


def _julia() -> bool:
    return izh._ensure_julia_loaded()


def _go() -> bool:
    return izh._ensure_go_loaded()


def _mojo() -> bool:
    return izh._ensure_mojo_loaded()


_BIT_EXACT = [("rust", _rust), ("julia", _julia), ("go", _go)]
_CURRENTS = [0.0, 100.0, 300.0, 500.0]


# ───────────────────── bit-exact backends (rust/julia/go) ─────────────────────


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
@pytest.mark.parametrize("current", _CURRENTS)
def test_bit_exact_trace(backend: str, available, current: float) -> None:
    if not available():
        pytest.skip(f"{backend} Izhikevich2007 backend unavailable")
    ref_trace, ref_spikes, rv, ru = _run("python", current=current)
    trace, spikes, vf, uf = _run(backend, current=current)
    np.testing.assert_array_equal(trace, ref_trace)
    assert spikes == ref_spikes
    assert (vf, uf) == (rv, ru)


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
def test_bit_exact_empty_and_single(backend: str, available) -> None:
    if not available():
        pytest.skip(f"{backend} Izhikevich2007 backend unavailable")
    for n in (0, 1, 2):
        ref, rs, rv, ru = _run("python", n=n)
        got, gs, gv, gu = _run(backend, n=n)
        np.testing.assert_array_equal(got, ref)
        assert (gs, gv, gu) == (rs, rv, ru)


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
def test_bit_exact_tonic_long_run(backend: str, available) -> None:
    # A long tonic-spiking run with many threshold resets stays bit-exact — the
    # exact RHS and the reset have no order sensitivity.
    if not available():
        pytest.skip(f"{backend} Izhikevich2007 backend unavailable")
    ref, rs, rv, ru = _run("python", current=300.0, n=60000)
    got, gs, gv, gu = _run(backend, current=300.0, n=60000)
    np.testing.assert_array_equal(got, ref)
    assert (gs, gv, gu) == (rs, rv, ru)


# ───────────────────────────── Mojo (FMA, ULP-bounded) ─────────────────────────


@pytest.mark.skipif(not _mojo(), reason="Mojo Izhikevich2007 backend unavailable")
@pytest.mark.parametrize("current", _CURRENTS)
def test_mojo_trace_ulp_bounded_and_exact_spikes(current: float) -> None:
    ref, ref_spikes, _rv, _ru = _run("python", current=current)
    got, spikes, _vf, _uf = _run("mojo", current=current)
    np.testing.assert_allclose(got, ref, atol=_MOJO_ATOL, rtol=0.0)
    assert spikes == ref_spikes


@pytest.mark.skipif(not _mojo(), reason="Mojo Izhikevich2007 backend unavailable")
def test_mojo_band_does_not_amplify() -> None:
    ref, ref_spikes, _rv, _ru = _run("python", current=300.0, n=50000)
    got, spikes, _vf, _uf = _run("mojo", current=300.0, n=50000)
    assert float(np.max(np.abs(got - ref))) < 1e-9
    assert spikes == ref_spikes


# ───────────────────────────── dispatch + algorithm ───────────────────────────


def test_auto_matches_python_bit_exact() -> None:
    ref, ref_spikes, _rv, _ru = _run("python")
    got, spikes, _vf, _uf = _run("auto")
    np.testing.assert_array_equal(got, ref)
    assert spikes == ref_spikes


def test_invalid_backend_raises() -> None:
    with pytest.raises(ValueError, match="backend must be"):
        Izhikevich2007Neuron().simulate(10, 0.0, backend="cuda")


def test_negative_n_steps_raises() -> None:
    with pytest.raises(ValueError, match="n_steps must be non-negative"):
        Izhikevich2007Neuron().simulate(-1, 0.0)


def test_non_finite_current_raises() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        Izhikevich2007Neuron().simulate(10, np.inf)


def test_non_rk4_integrator_rejected() -> None:
    with pytest.raises(ValueError, match="RK4 integrator only"):
        Izhikevich2007Neuron(integrator="euler").simulate(10, 300.0)


def test_simulate_matches_repeated_step() -> None:
    trace_a, spikes_a = Izhikevich2007Neuron().simulate(500, 300.0, backend="python")
    manual = []
    spikes_b = 0
    stepper = Izhikevich2007Neuron()
    for _ in range(500):
        spikes_b += stepper.step(300.0)
        manual.append(stepper.v)
    np.testing.assert_array_equal(trace_a, np.asarray(manual, dtype=np.float64))
    assert spikes_a == spikes_b


def test_final_state_advances_instance() -> None:
    neuron = Izhikevich2007Neuron()
    _trace, _spikes = neuron.simulate(500, 300.0, backend="python")
    manual = Izhikevich2007Neuron()
    for _ in range(500):
        manual.step(300.0)
    assert (neuron.v, neuron.u) == (manual.v, manual.u)


def test_tonic_firing_under_drive() -> None:
    _trace, spikes = Izhikevich2007Neuron().simulate(20000, 300.0, backend="python")
    assert spikes > 10


def test_subthreshold_silent() -> None:
    _trace, spikes = Izhikevich2007Neuron().simulate(20000, 0.0, backend="python")
    assert spikes == 0


def test_trace_resets_below_vpeak() -> None:
    # Every recorded sample is at or below the peak (the reset fires on >= vpeak).
    trace, spikes = Izhikevich2007Neuron().simulate(20000, 300.0, backend="python")
    assert spikes > 0
    assert np.all(trace <= Izhikevich2007Neuron().vpeak)
