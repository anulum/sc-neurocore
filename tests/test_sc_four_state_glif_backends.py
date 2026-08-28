# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Polyglot parity tests for the retained four-state GLIF neuron

"""Cross-backend parity for ``SCFourStateGLIFNeuron.simulate``.

The retained four-state project recurrence has a purely linear
right-hand side (additions, multiplications, divisions; no transcendental
functions) advanced by candidate-first RK4 with a discontinuous spike reset.
Because every stage is exact arithmetic, the Rust engine, Julia and Go backends
reproduce the pure-NumPy reference bit-for-bit — trace, spike count and the final
``(v, theta, i_asc1, i_asc2)`` state all match exactly. The Mojo backend fuses
multiply-add, so it is validated as non-amplifying within a tight ULP band with
identical spike counts rather than bit-exactly (the band is compiler/version
dependent and must not be promoted to a strict equality assertion).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pytest
from numpy.typing import NDArray

from sc_neurocore.neurons.models import sc_four_state_glif as glif
from sc_neurocore.neurons.models.sc_four_state_glif import SCFourStateGLIFNeuron


def _run(
    backend: str,
    *,
    n: int = 4000,
    current: float = 30.0,
    parameters: Mapping[str, float] | None = None,
) -> tuple[NDArray[np.float64], int, tuple[float, float, float, float]]:
    neuron = SCFourStateGLIFNeuron(**dict(parameters or {}))
    trace, spikes = neuron.simulate(n, current, backend=backend)
    return trace, spikes, (neuron.v, neuron.theta, neuron.i_asc1, neuron.i_asc2)


def _rust() -> bool:
    return glif._HAS_RUST


def _julia() -> bool:
    return glif._ensure_julia_loaded()


def _go() -> bool:
    return glif._ensure_go_loaded()


def _mojo() -> bool:
    return glif._ensure_mojo_loaded()


_BIT_EXACT: tuple[tuple[str, Callable[[], bool]], ...] = (
    ("rust", _rust),
    ("julia", _julia),
    ("go", _go),
)
_CURRENTS: tuple[float, ...] = (0.0, 22.0, 30.0, 45.0)
_REGIMES: tuple[dict[str, float], ...] = (
    dict(),
    dict(a_theta=0.05, delta_theta=5.0, r_asc1=2.0, r_asc2=1.0),
    dict(tau_m=5.0, tau_theta=50.0, a_theta=0.02, resistance=2.0),
    dict(delta_theta=0.0, r_asc1=0.0, r_asc2=0.0, a_theta=0.0),
)


# ───────────── Rust / Julia / Go: bit-exact (linear exact arithmetic) ──────────


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
@pytest.mark.parametrize("current", _CURRENTS)
def test_bit_exact_currents(backend: str, available: Callable[[], bool], current: float) -> None:
    if not available():
        pytest.skip(f"{backend} SC four-state GLIF backend unavailable")
    ref_trace, ref_spikes, ref_state = _run("python", current=current)
    trace, spikes, state = _run(backend, current=current)
    np.testing.assert_array_equal(trace, ref_trace)
    assert spikes == ref_spikes
    assert state == ref_state


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
@pytest.mark.parametrize("regime", _REGIMES, ids=["r0", "r1", "r2", "r3"])
def test_bit_exact_regimes(
    backend: str, available: Callable[[], bool], regime: dict[str, float]
) -> None:
    if not available():
        pytest.skip(f"{backend} SC four-state GLIF backend unavailable")
    ref_trace, ref_spikes, ref_state = _run("python", current=30.0, parameters=regime)
    trace, spikes, state = _run(backend, current=30.0, parameters=regime)
    np.testing.assert_array_equal(trace, ref_trace)
    assert spikes == ref_spikes
    assert state == ref_state


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
def test_bit_exact_empty_and_long_horizon(backend: str, available: Callable[[], bool]) -> None:
    if not available():
        pytest.skip(f"{backend} SC four-state GLIF backend unavailable")
    for n in (0, 1, 2, 60_000):
        ref, rs, rstate = _run("python", n=n)
        got, gs, gstate = _run(backend, n=n)
        np.testing.assert_array_equal(got, ref)
        assert (gs, gstate) == (rs, rstate)


# ───────────────────── Mojo: non-amplifying (FMA fusion) ───────────────────────


@pytest.mark.parametrize("current", _CURRENTS)
def test_mojo_non_amplifying_currents(current: float) -> None:
    if not _mojo():
        pytest.skip("mojo SC four-state GLIF backend unavailable")
    ref, ref_spikes, _state = _run("python", n=20_000, current=current)
    got, spikes, _gstate = _run("mojo", n=20_000, current=current)
    np.testing.assert_allclose(got, ref, atol=1e-9, rtol=0.0)
    assert spikes == ref_spikes


@pytest.mark.parametrize("regime", _REGIMES, ids=["r0", "r1", "r2", "r3"])
def test_mojo_non_amplifying_regimes(regime: dict[str, float]) -> None:
    if not _mojo():
        pytest.skip("mojo SC four-state GLIF backend unavailable")
    # The gap at 60k steps must remain at the ULP level (no amplification).
    ref, ref_spikes, _state = _run("python", n=60_000, current=30.0, parameters=regime)
    got, spikes, _gstate = _run("mojo", n=60_000, current=30.0, parameters=regime)
    assert float(np.max(np.abs(got - ref))) < 1e-8
    assert spikes == ref_spikes


# ───────────────────────────── dispatch + algorithm ───────────────────────────


def test_auto_matches_python() -> None:
    ref, ref_spikes, ref_state = _run("python")
    got, spikes, state = _run("auto")
    np.testing.assert_allclose(got, ref, atol=1e-9, rtol=0.0)
    assert spikes == ref_spikes
    assert state == ref_state


def test_invalid_backend_raises() -> None:
    with pytest.raises(ValueError, match="backend must be"):
        SCFourStateGLIFNeuron().simulate(10, 0.0, backend="cuda")


def test_negative_n_steps_raises() -> None:
    with pytest.raises(ValueError, match="n_steps must be between"):
        SCFourStateGLIFNeuron().simulate(-1, 0.0)


def test_non_finite_current_raises() -> None:
    with pytest.raises(ValueError, match="current must be finite"):
        SCFourStateGLIFNeuron().simulate(10, float("inf"))


def test_invalid_runtime_raises() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        SCFourStateGLIFNeuron(tau_m=-1.0).simulate(10, 30.0)


def test_simulate_matches_repeated_step() -> None:
    trace_a, spikes_a = SCFourStateGLIFNeuron().simulate(400, 30.0, backend="python")
    manual = []
    spikes_b = 0
    stepper = SCFourStateGLIFNeuron()
    for _ in range(400):
        spikes_b += stepper.step(30.0)
        manual.append(stepper.v)
    np.testing.assert_array_equal(trace_a, np.asarray(manual, dtype=np.float64))
    assert spikes_a == spikes_b
