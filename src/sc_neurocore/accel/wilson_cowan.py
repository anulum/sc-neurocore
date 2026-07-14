# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wilson-Cowan acceleration loading and result validation

"""Dispatch the complete Wilson-Cowan batch contract across five runtimes."""

from __future__ import annotations

import importlib
import math
from typing import Any, Protocol, SupportsFloat, cast

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel.backend_order import with_floor
from sc_neurocore.accel.backend_selection import select_backend_order

WilsonCowanResult = tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    float,
    float,
]
KERNEL = "wilson_cowan_rk4_batch"
PARITY_ATOL = {
    "python": 0.0,
    "rust": 1.0e-9,
    "julia": 1.0e-9,
    "go": 1.0e-9,
    "mojo": 1.0e-8,
}
_AUTO_BACKENDS = with_floor("python")
_RESULT_TOLERANCE = 1.0e-9


def _logistic(value: float) -> float:
    """Return a branch-stable scalar logistic value."""
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


class _EngineRunner(Protocol):
    """Typed PyO3 Wilson-Cowan batch boundary."""

    def __call__(
        self,
        e_init: float,
        i_init: float,
        w_ee: float,
        w_ei: float,
        w_ie: float,
        w_ii: float,
        tau_e: float,
        tau_i: float,
        a: float,
        theta: float,
        dt: float,
        ext_input: npt.NDArray[np.float64],
    ) -> dict[str, object]: ...


def _load_engine_runner() -> _EngineRunner:
    """Return the installed Rust engine batch function."""
    engine = importlib.import_module("sc_neurocore_engine")
    return cast(_EngineRunner, engine.py_wilson_cowan_simulate)


try:
    _engine_simulate: _EngineRunner | None = _load_engine_runner()
    _HAS_RUST = True
except (ImportError, AttributeError):
    _engine_simulate = None
    _HAS_RUST = False


def _module(name: str) -> Any:
    """Import one existing language-specific Wilson-Cowan facade."""
    return importlib.import_module(f"sc_neurocore.accel.{name}.wilson_cowan")


def backend_available(backend: str) -> bool:
    """Return whether one public execution lane is ready."""
    if backend == "rust":
        return _HAS_RUST and _engine_simulate is not None
    if backend == "julia":
        try:
            module = importlib.import_module("sc_neurocore.accel.julia.neurons")
            module._ensure_wilson_cowan_loaded()
        except (ImportError, FileNotFoundError, RuntimeError):
            return False
        return True
    if backend in {"go", "mojo"}:
        try:
            module = _module(backend)
        except ImportError:
            return False
        marker = f"_HAS_{backend.upper()}_WILSON_COWAN"
        return bool(getattr(module, marker, False))
    return backend == "python"


def auto_backend() -> str:
    """Choose the first available lane from committed measured evidence."""
    ordered = select_backend_order(KERNEL, static=_AUTO_BACKENDS)
    return next((backend for backend in ordered if backend_available(backend)), "python")


def _float(value: object, name: str) -> float:
    """Convert one backend scalar into a finite float."""
    try:
        converted = float(cast(SupportsFloat, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise FloatingPointError(f"WilsonCowan backend returned non-numeric {name}.") from exc
    if not np.isfinite(converted):
        raise FloatingPointError(f"WilsonCowan backend returned non-finite {name}.")
    return converted


def normalise_result(
    e_trace: npt.ArrayLike,
    i_trace: npt.ArrayLike,
    e_final: object,
    i_final: object,
    *,
    n_steps: int,
    initial_e: float,
    initial_i: float,
    a: float,
    theta: float,
) -> WilsonCowanResult:
    """Reject malformed or non-atomic backend output before public commit."""
    try:
        e_values = np.asarray(e_trace, dtype=np.float64)
        i_values = np.asarray(i_trace, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FloatingPointError("WilsonCowan backend returned non-numeric traces.") from exc
    if e_values.ndim != 1 or e_values.shape != (n_steps,):
        raise FloatingPointError("WilsonCowan backend returned a malformed E trace.")
    if i_values.ndim != 1 or i_values.shape != (n_steps,):
        raise FloatingPointError("WilsonCowan backend returned a malformed I trace.")
    if not np.isfinite(e_values).all() or not np.isfinite(i_values).all():
        raise FloatingPointError("WilsonCowan backend returned non-finite rates.")

    baseline = _logistic(-a * theta)
    lower = -baseline - _RESULT_TOLERANCE
    upper = 1.0 + _RESULT_TOLERANCE
    if not np.logical_and(e_values >= lower, e_values <= upper).all():
        raise FloatingPointError("WilsonCowan backend returned an out-of-range E rate.")
    if not np.logical_and(i_values >= lower, i_values <= upper).all():
        raise FloatingPointError("WilsonCowan backend returned an out-of-range I rate.")

    final_e = _float(e_final, "final E rate")
    final_i = _float(i_final, "final I rate")
    expected_e = initial_e if n_steps == 0 else float(e_values[-1])
    expected_i = initial_i if n_steps == 0 else float(i_values[-1])
    if abs(final_e - expected_e) > _RESULT_TOLERANCE:
        raise FloatingPointError("WilsonCowan final E rate disagrees with its trace.")
    if abs(final_i - expected_i) > _RESULT_TOLERANCE:
        raise FloatingPointError("WilsonCowan final I rate disagrees with its trace.")
    if not lower <= final_e <= upper or not lower <= final_i <= upper:
        raise FloatingPointError("WilsonCowan backend returned invalid final rates.")
    return (
        np.ascontiguousarray(e_values, dtype=np.float64),
        np.ascontiguousarray(i_values, dtype=np.float64),
        final_e,
        final_i,
    )


def _normalise_mapping(
    result: dict[str, object],
    *,
    n_steps: int,
    initial_e: float,
    initial_i: float,
    a: float,
    theta: float,
) -> WilsonCowanResult:
    """Validate one mapping returned by an existing native facade."""
    try:
        e_trace = cast(npt.ArrayLike, result["e"])
        i_trace = cast(npt.ArrayLike, result["i"])
        e_final = result["e_final"]
        i_final = result["i_final"]
    except KeyError as exc:
        raise FloatingPointError("WilsonCowan backend returned an incomplete result.") from exc
    return normalise_result(
        e_trace,
        i_trace,
        e_final,
        i_final,
        n_steps=n_steps,
        initial_e=initial_e,
        initial_i=initial_i,
        a=a,
        theta=theta,
    )


def _arguments(
    e: float,
    i: float,
    w_ee: float,
    w_ei: float,
    w_ie: float,
    w_ii: float,
    tau_e: float,
    tau_i: float,
    a: float,
    theta: float,
    dt: float,
    n_steps: int,
    current: float,
) -> tuple[object, ...]:
    """Build the shared scalar-plus-drive native call contract."""
    ext_input = np.full(n_steps, current, dtype=np.float64)
    return (e, i, w_ee, w_ei, w_ie, w_ii, tau_e, tau_i, a, theta, dt, ext_input)


def _simulate(
    runner: Any,
    e: float,
    i: float,
    w_ee: float,
    w_ei: float,
    w_ie: float,
    w_ii: float,
    tau_e: float,
    tau_i: float,
    a: float,
    theta: float,
    dt: float,
    n_steps: int,
    current: float,
) -> WilsonCowanResult:
    """Run and normalise one mapping-returning native implementation."""
    result = runner(
        *_arguments(
            e,
            i,
            w_ee,
            w_ei,
            w_ie,
            w_ii,
            tau_e,
            tau_i,
            a,
            theta,
            dt,
            n_steps,
            current,
        )
    )
    return _normalise_mapping(
        result,
        n_steps=n_steps,
        initial_e=e,
        initial_i=i,
        a=a,
        theta=theta,
    )


def simulate_rust(
    e: float,
    i: float,
    w_ee: float,
    w_ei: float,
    w_ie: float,
    w_ii: float,
    tau_e: float,
    tau_i: float,
    a: float,
    theta: float,
    dt: float,
    n_steps: int,
    current: float,
) -> WilsonCowanResult:
    """Run the complete contract through the production Rust engine."""
    if _engine_simulate is None:
        raise RuntimeError("Rust WilsonCowan backend is unavailable.")
    return _simulate(
        _engine_simulate,
        e,
        i,
        w_ee,
        w_ei,
        w_ie,
        w_ii,
        tau_e,
        tau_i,
        a,
        theta,
        dt,
        n_steps,
        current,
    )


def simulate_julia(
    e: float,
    i: float,
    w_ee: float,
    w_ei: float,
    w_ie: float,
    w_ii: float,
    tau_e: float,
    tau_i: float,
    a: float,
    theta: float,
    dt: float,
    n_steps: int,
    current: float,
) -> WilsonCowanResult:
    """Run the Julia recurrence through its JuliaCall facade."""
    module = importlib.import_module("sc_neurocore.accel.julia.neurons")
    return _simulate(
        module.simulate_wilson_cowan,
        e,
        i,
        w_ee,
        w_ei,
        w_ie,
        w_ii,
        tau_e,
        tau_i,
        a,
        theta,
        dt,
        n_steps,
        current,
    )


def simulate_go(
    e: float,
    i: float,
    w_ee: float,
    w_ei: float,
    w_ie: float,
    w_ii: float,
    tau_e: float,
    tau_i: float,
    a: float,
    theta: float,
    dt: float,
    n_steps: int,
    current: float,
) -> WilsonCowanResult:
    """Run the Go recurrence through its generated C ABI."""
    module = _module("go")
    return _simulate(
        module.simulate_wilson_cowan,
        e,
        i,
        w_ee,
        w_ei,
        w_ie,
        w_ii,
        tau_e,
        tau_i,
        a,
        theta,
        dt,
        n_steps,
        current,
    )


def simulate_mojo(
    e: float,
    i: float,
    w_ee: float,
    w_ei: float,
    w_ie: float,
    w_ii: float,
    tau_e: float,
    tau_i: float,
    a: float,
    theta: float,
    dt: float,
    n_steps: int,
    current: float,
) -> WilsonCowanResult:
    """Run the Mojo recurrence through its exported C ABI."""
    module = _module("mojo")
    return _simulate(
        module.simulate_wilson_cowan,
        e,
        i,
        w_ee,
        w_ei,
        w_ie,
        w_ii,
        tau_e,
        tau_i,
        a,
        theta,
        dt,
        n_steps,
        current,
    )
