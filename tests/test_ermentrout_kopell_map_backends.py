# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Polyglot parity tests for the Ermentrout-Kopell theta map

"""Cross-backend parity for ``ErmentroutKopellMapNeuron.simulate``.

The only transcendental is ``cos`` and the theta neuron is a non-chaotic phase
oscillator (Lyapunov exponent 0), so floating-point differences do not amplify.
On a shared libm the Rust backend reproduces the NumPy reference **bit-for-bit**
(Python ``math.cos`` and Rust ``f64::cos`` resolve to the same glibc symbol, and
the in-regime wrap is identical). Julia, Go and Mojo use their own ``cos`` and so
differ by a small, **non-amplifying** ULP band (measured well under 1e-9 over the
whole trace), but every backend produces the **same spike count** — threshold
crossings of ``pi`` are robust to those sub-ULP phase perturbations.
"""

from __future__ import annotations

from collections.abc import Callable
import ctypes
import math

import numpy as np
import pytest

from sc_neurocore.neurons.models import ermentrout_kopell_map_neuron as ek
from sc_neurocore.neurons.models.ermentrout_kopell_map_neuron import ErmentroutKopellMapNeuron
from tests.julia_requirement import require_julia

# Non-amplifying ULP band for the libm-divergent backends (measured ~7e-14).
_TRACE_ATOL = 1e-9


def _run(
    backend: str, *, theta0: float = 0.0, n: int = 4000, current: float = 0.1
) -> tuple[np.ndarray, int, float]:
    neuron = ErmentroutKopellMapNeuron(theta=theta0)
    trace, spikes = neuron.simulate(n, current, backend=backend)
    return trace, spikes, neuron.theta


def _rust() -> bool:
    return ek._HAS_RUST


def _julia() -> bool:
    return ek._ensure_julia_loaded()


def _go() -> bool:
    return ek._ensure_go_loaded()


def _mojo() -> bool:
    return ek._ensure_mojo_loaded()


_ULP_BOUNDED = [("julia", _julia), ("go", _go), ("mojo", _mojo)]
_CURRENTS = [0.05, 0.1, 0.5, 1.0]


# ───────────────────────── Rust (bit-exact, shared libm) ──────────────────────


@pytest.mark.skipif(not _rust(), reason="Rust Ermentrout-Kopell backend unavailable")
@pytest.mark.parametrize("current", _CURRENTS)
def test_rust_bit_exact(current: float) -> None:
    """The Rust lane reproduces the Python trace and spike count bit for bit at each drive."""
    ref_trace, ref_spikes, rt = _run("python", current=current)
    trace, spikes, tf = _run("rust", current=current)
    np.testing.assert_array_equal(trace, ref_trace)
    assert spikes == ref_spikes
    assert tf == rt


@pytest.mark.skipif(not _rust(), reason="Rust Ermentrout-Kopell backend unavailable")
def test_rust_bit_exact_empty_and_single() -> None:
    """The Rust lane matches Python exactly for zero, one and two steps."""
    for n in (0, 1, 2):
        ref, rs, rt = _run("python", n=n)
        got, gs, gt = _run("rust", n=n)
        np.testing.assert_array_equal(got, ref)
        assert (gs, gt) == (rs, rt)


# ──────────────── Julia / Go / Mojo (own libm cos, ULP-bounded) ───────────────


@pytest.mark.parametrize("backend,available", _ULP_BOUNDED, ids=[b for b, _ in _ULP_BOUNDED])
@pytest.mark.parametrize("current", _CURRENTS)
def test_ulp_bounded_trace_and_exact_spikes(
    backend: str, available: Callable[[], bool], current: float
) -> None:
    """Each native lane keeps the trace within the ULP bound and the spike count exact."""
    if not available():
        pytest.skip(f"{backend} Ermentrout-Kopell backend unavailable")
    ref_trace, ref_spikes, _rt = _run("python", current=current)
    trace, spikes, _tf = _run(backend, current=current)
    # Phase trace stays within a non-amplifying ULP band of the reference...
    np.testing.assert_allclose(trace, ref_trace, atol=_TRACE_ATOL, rtol=0.0)
    # ...and the coarse spike count is identical (threshold crossings are robust).
    assert spikes == ref_spikes


@pytest.mark.parametrize("backend,available", _ULP_BOUNDED, ids=[b for b, _ in _ULP_BOUNDED])
def test_ulp_band_does_not_amplify(backend: str, available: Callable[[], bool]) -> None:
    # Over a long horizon the divergence must stay bounded (non-chaotic): a
    # chaotic map would blow this past O(1).
    """Over a long horizon a native lane's divergence from Python stays bounded."""
    if not available():
        pytest.skip(f"{backend} Ermentrout-Kopell backend unavailable")
    ref, _rs, _rt = _run("python", n=50000, current=0.3)
    got, _gs, _gt = _run(backend, n=50000, current=0.3)
    assert float(np.max(np.abs(got - ref))) < 1e-9


@pytest.mark.parametrize("backend,available", _ULP_BOUNDED, ids=[b for b, _ in _ULP_BOUNDED])
def test_ulp_bounded_empty_and_single(backend: str, available: Callable[[], bool]) -> None:
    """Each native lane stays within the ULP bound for zero, one and two steps."""
    if not available():
        pytest.skip(f"{backend} Ermentrout-Kopell backend unavailable")
    for n in (0, 1, 2):
        ref, rs, _rt = _run("python", n=n)
        got, gs, _gt = _run(backend, n=n)
        np.testing.assert_allclose(got, ref, atol=_TRACE_ATOL, rtol=0.0)
        assert gs == rs


# ───────────────────────────── dispatch + algorithm ───────────────────────────


def test_auto_matches_python_bit_exact() -> None:
    # auto -> Rust, which is bit-exact on the shared libm.
    """The automatic backend, which selects Rust, matches Python bit for bit."""
    ref, ref_spikes, _rt = _run("python")
    got, spikes, _tf = _run("auto")
    np.testing.assert_array_equal(got, ref)
    assert spikes == ref_spikes


def test_invalid_backend_raises() -> None:
    """An unknown backend name is refused."""
    with pytest.raises(ValueError, match="backend must be"):
        ErmentroutKopellMapNeuron().simulate(10, 0.0, backend="cuda")


def test_negative_n_steps_raises() -> None:
    """A negative step count is refused."""
    with pytest.raises(ValueError, match="n_steps must be between"):
        ErmentroutKopellMapNeuron().simulate(-1, 0.0)


@pytest.mark.parametrize("n_steps", (True, 1.5, 1 << 31))
def test_invalid_step_count_raises(n_steps: object) -> None:
    """A step count that is not a valid integer is refused."""
    with pytest.raises(ValueError, match="n_steps"):
        ErmentroutKopellMapNeuron().simulate(n_steps, 0.0)  # type: ignore[arg-type]


def test_non_finite_current_raises() -> None:
    """A non-finite drive current is refused."""
    with pytest.raises(ValueError, match="current must be finite"):
        ErmentroutKopellMapNeuron().simulate(10, np.inf)


def test_simulate_matches_repeated_step() -> None:
    """One simulate call gives the same trace and spikes as repeated steps."""
    trace_a, spikes_a = ErmentroutKopellMapNeuron().simulate(500, 0.1, backend="python")
    manual = []
    spikes_b = 0
    stepper = ErmentroutKopellMapNeuron()
    for _ in range(500):
        spikes_b += stepper.step(0.1)
        manual.append(stepper.theta)
    np.testing.assert_array_equal(trace_a, np.asarray(manual, dtype=np.float64))
    assert spikes_a == spikes_b


def test_final_state_advances_instance() -> None:
    """After simulate the instance holds the state that repeated steps reach."""
    neuron = ErmentroutKopellMapNeuron()
    _trace, _spikes = neuron.simulate(500, 0.1, backend="python")
    manual = ErmentroutKopellMapNeuron()
    for _ in range(500):
        manual.step(0.1)
    assert neuron.theta == manual.theta


def test_drive_produces_spikes_monotonic() -> None:
    """A stronger drive never produces fewer spikes."""
    counts = []
    for current in (0.05, 0.1, 0.5, 1.0):
        _trace, spikes = ErmentroutKopellMapNeuron().simulate(20000, current, backend="python")
        counts.append(spikes)
    assert counts[0] > 0
    assert all(counts[i] <= counts[i + 1] for i in range(len(counts) - 1))


def test_zero_current_rests_at_origin() -> None:
    # theta0 = 0 with no drive is the stable rest phase (1 - cos 0 = 0).
    """With no drive the phase rests at zero and the neuron never spikes."""
    trace, spikes = ErmentroutKopellMapNeuron().simulate(1000, 0.0, backend="python")
    assert spikes == 0
    assert np.all(trace == 0.0)


def test_phase_confined_to_circle() -> None:
    """The phase stays in [0, 2 pi) over a long driven run."""
    trace, _spikes = ErmentroutKopellMapNeuron().simulate(50000, 0.5, backend="python")
    two_pi = 2.0 * math.pi
    assert np.all(trace >= 0.0) and np.all(trace < two_pi)


@pytest.mark.parametrize(
    ("backend", "available"),
    (("python", lambda: True), ("rust", _rust), ("julia", _julia), ("go", _go), ("mojo", _mojo)),
)
def test_public_dispatch_rejects_mutated_configuration_atomically(
    backend: str, available: Callable[[], bool]
) -> None:
    """A configuration made invalid after construction is refused without changing the state."""
    if not available():
        pytest.skip(f"{backend} Ermentrout-Kopell backend unavailable")
    neuron = ErmentroutKopellMapNeuron(theta=0.25)
    neuron.dt = 0.0
    with pytest.raises(ValueError, match="dt must be positive"):
        neuron.simulate(1, 0.1, backend=backend)
    assert neuron.theta == 0.25


def test_go_c_abi_rejects_invalid_input_without_writing_trace() -> None:
    """The Go C ABI refuses invalid input and leaves the caller's trace buffer untouched."""
    if not _go():
        pytest.skip("Go Ermentrout-Kopell backend unavailable")
    trace = np.full(2, 123.0, dtype=np.float64)
    assert ek._go_lib is not None
    result = ek._go_lib.ermentrout_kopell_map_simulate_c(
        ctypes.c_double(0.0),
        ctypes.c_double(0.1),
        ctypes.c_double(1.0),
        ctypes.c_double(math.pi),
        ctypes.c_int(1),
        ctypes.c_double(math.nan),
        trace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    assert result == -1
    np.testing.assert_array_equal(trace, np.array([123.0, 123.0]))


def test_mojo_c_abi_rejects_invalid_input_without_writing_trace() -> None:
    """The Mojo C ABI refuses invalid input and leaves the caller's trace buffer untouched."""
    if not _mojo():
        pytest.skip("Mojo Ermentrout-Kopell backend unavailable")
    trace = np.full(2, 123.0, dtype=np.float64)
    assert ek._mojo_lib is not None
    result = ek._mojo_lib.ermentrout_kopell_map_simulate_c(
        0.0,
        0.1,
        1.0,
        math.pi,
        1,
        math.nan,
        int(trace.ctypes.data),
    )
    assert result == -1
    np.testing.assert_array_equal(trace, np.array([123.0, 123.0]))


def test_julia_native_api_rejects_invalid_input() -> None:
    """The Julia kernel refuses a non-finite input with a Julia error."""
    juliacall = require_julia()
    assert _julia(), "the Julia Ermentrout-Kopell kernel did not load with a healthy bridge"
    with pytest.raises(juliacall.JuliaError, match="inputs must be finite"):
        assert ek._julia_module is not None
        ek._julia_module.simulate_trace(0.0, 0.1, 1.0, math.pi, 1, math.nan)
