# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Polyglot parity tests for the Terman-Wang 1995 LEGION oscillator

"""Cross-backend parity for ``TermanWangOscillator.simulate``.

The right-hand side mixes an exact cubic (`v*v*v`, matching the engine's
`v.powi(3)`) with a ``tanh`` gate. Rust resolves the same host ``tanh`` as
Python. Julia, Go, and Mojo use their own math libraries, so this suite bounds
their complete enrolled traces and requires exact event counts.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.neurons.models import terman_wang
from sc_neurocore.neurons.models.terman_wang import TermanWangOscillator


def _run(
    backend: str, *, n: int = 4000, current: float = 0.5, **params: float
) -> tuple[npt.NDArray[np.float64], int, float, float]:
    neuron = TermanWangOscillator(**params)
    trace, spikes = neuron.simulate(n, current, backend=backend)
    return trace, spikes, neuron.v, neuron.w


def _rust() -> bool:
    return terman_wang._HAS_RUST


def _julia() -> bool:
    return terman_wang._ensure_julia_loaded()


def _go() -> bool:
    return terman_wang._ensure_go_loaded()


def _mojo() -> bool:
    return terman_wang._ensure_mojo_loaded()


_ULP_BOUNDED = [("julia", _julia), ("go", _go), ("mojo", _mojo)]
_CURRENTS = [0.0, 0.5, 1.0, 1.5]


class _Regime(TypedDict, total=False):
    alpha: float
    beta: float
    epsilon: float
    rho: float


_REGIMES: list[_Regime] = [
    dict(alpha=3.0, beta=0.2, epsilon=0.02),
    dict(alpha=2.0, beta=0.3, epsilon=0.04),
    dict(alpha=4.0, beta=0.15, epsilon=0.01, rho=0.1),
]


# ───────────────────── Rust: bit-exact (shared glibc tanh) ─────────────────────


@pytest.mark.skipif(not _rust(), reason="Rust Terman-Wang backend unavailable")
@pytest.mark.parametrize("current", _CURRENTS)
def test_rust_bit_exact_currents(current: float) -> None:
    ref_trace, ref_spikes, rv, rw = _run("python", current=current)
    trace, spikes, vf, wf = _run("rust", current=current)
    np.testing.assert_array_equal(trace, ref_trace)
    assert spikes == ref_spikes
    assert vf == rv and wf == rw


@pytest.mark.skipif(not _rust(), reason="Rust Terman-Wang backend unavailable")
@pytest.mark.parametrize("regime", _REGIMES, ids=["r0", "r1", "r2"])
def test_rust_bit_exact_regimes(regime: _Regime) -> None:
    ref_trace, ref_spikes, rv, rw = _run("python", current=0.5, **regime)
    trace, spikes, vf, wf = _run("rust", current=0.5, **regime)
    np.testing.assert_array_equal(trace, ref_trace)
    assert spikes == ref_spikes
    assert vf == rv and wf == rw


@pytest.mark.skipif(not _rust(), reason="Rust Terman-Wang backend unavailable")
def test_rust_bit_exact_empty_and_long_horizon() -> None:
    for n in (0, 1, 2, 60_000):
        ref, rs, rv, rw = _run("python", n=n)
        got, gs, gv, gw = _run("rust", n=n)
        np.testing.assert_array_equal(got, ref)
        assert (gs, gv, gw) == (rs, rv, rw)


# ───────────────── Julia / Go / Mojo: ULP-bounded (own libm tanh) ──────────────


@pytest.mark.parametrize("backend,available", _ULP_BOUNDED, ids=[b for b, _ in _ULP_BOUNDED])
@pytest.mark.parametrize("current", _CURRENTS)
def test_ulp_bounded_whole_trace(
    backend: str, available: Callable[[], bool], current: float
) -> None:
    if not available():
        pytest.skip(f"{backend} Terman-Wang backend unavailable")
    ref, ref_spikes, _rv, _rw = _run("python", n=20_000, current=current)
    got, spikes, _vf, _wf = _run(backend, n=20_000, current=current)
    np.testing.assert_allclose(got, ref, atol=1e-9, rtol=0.0)
    assert spikes == ref_spikes


@pytest.mark.parametrize("backend,available", _ULP_BOUNDED, ids=[b for b, _ in _ULP_BOUNDED])
def test_ulp_bounded_enrolled_long_horizon(backend: str, available: Callable[[], bool]) -> None:
    if not available():
        pytest.skip(f"{backend} Terman-Wang backend unavailable")
    ref, _rs, _rv, _rw = _run("python", n=60_000, current=0.5)
    got, _gs, _gv, _gw = _run(backend, n=60_000, current=0.5)
    assert float(np.max(np.abs(got - ref))) < 1e-8


# ───────────────────────────── dispatch + algorithm ───────────────────────────


def test_auto_matches_python() -> None:
    ref, ref_spikes, _rv, _rw = _run("python")
    got, spikes, _vf, _wf = _run("auto")
    np.testing.assert_allclose(got, ref, atol=1e-9, rtol=0.0)
    assert spikes == ref_spikes


def test_invalid_backend_raises() -> None:
    with pytest.raises(ValueError, match="backend must be"):
        TermanWangOscillator().simulate(10, 0.0, backend="cuda")


def test_negative_n_steps_raises() -> None:
    with pytest.raises(ValueError, match="n_steps must be non-negative"):
        TermanWangOscillator().simulate(-1, 0.0)


def test_non_finite_current_raises() -> None:
    with pytest.raises(FloatingPointError, match="must be finite"):
        TermanWangOscillator().simulate(10, np.inf)


def test_numeric_contract_validation_is_failure_atomic() -> None:
    neuron = TermanWangOscillator()
    object.__setattr__(neuron, "v", 1)
    neuron.epsilon = float("nan")

    with pytest.raises(FloatingPointError, match="epsilon must be finite"):
        neuron.simulate(1, 0.0, backend="python")

    assert type(neuron.v) is int


@pytest.mark.parametrize(
    ("backend", "available"),
    (("python", lambda: True), ("rust", _rust), ("julia", _julia), ("go", _go), ("mojo", _mojo)),
)
def test_batch_overflow_is_failure_atomic(backend: str, available: Callable[[], bool]) -> None:
    if not available():
        pytest.skip(f"{backend} Terman-Wang backend unavailable")
    neuron = TermanWangOscillator(v=1.0e103, w=-0.5)
    before = (neuron.v, neuron.w)

    with pytest.raises(FloatingPointError, match="invalid|overflow|non-finite|rejected"):
        neuron.simulate(2, 0.5, backend=backend)

    assert (neuron.v, neuron.w) == before


def test_simulate_matches_repeated_step() -> None:
    trace_a, spikes_a = TermanWangOscillator().simulate(400, 0.5, backend="python")
    manual = []
    spikes_b = 0
    stepper = TermanWangOscillator()
    for _ in range(400):
        spikes_b += stepper.step(0.5)
        manual.append(stepper.v)
    np.testing.assert_array_equal(trace_a, np.asarray(manual, dtype=np.float64))
    assert spikes_a == spikes_b
