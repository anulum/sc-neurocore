# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia-backed neuron kernels (juliacall dispatch)

"""Python entry points for Julia neuron kernels under this directory.

Each helper below lazily boots Julia via ``juliacall`` the first time it
is called and caches the compiled module so subsequent calls skip the
JIT warm-up (~5-10 s on cold start; sub-millisecond warm).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

try:
    from juliacall import JuliaError as _JuliacallError
    from juliacall import Main as _jl

    _JULIA_ERROR_TYPE: type[BaseException] | None = _JuliacallError
    _HAS_JULIA_NEURONS = True
except ImportError:
    _jl = None
    _JULIA_ERROR_TYPE = None
    _HAS_JULIA_NEURONS = False


_KERNEL_DIR = Path(__file__).resolve().parent
_ERMENTROUT_KOPELL_POP_LOADED = False
_RESONATE_AND_FIRE_LOADED = False
_WONG_WANG_LOADED = False
_JANSEN_RIT_LOADED = False
_WILSON_COWAN_LOADED = False
_RK4_NEURONS_LOADED = False


def is_julia_error(error: BaseException) -> bool:
    """Return whether ``error`` is the maintained Julia bridge exception."""
    return _JULIA_ERROR_TYPE is not None and isinstance(error, _JULIA_ERROR_TYPE)


def _ensure_wong_wang_loaded() -> Any:
    """Include `wong_wang.jl` into Julia Main on first use; return the module."""
    global _WONG_WANG_LOADED
    if _jl is None:
        raise ImportError("juliacall not available; install the `julia` extra")
    if not _WONG_WANG_LOADED:
        jl_path = _KERNEL_DIR / "wong_wang.jl"
        if not jl_path.is_file():
            raise FileNotFoundError(f"wong_wang.jl missing at {jl_path}")
        _jl.include(str(jl_path))
        _WONG_WANG_LOADED = True
    return _jl.WongWangAccel


def _ensure_jansen_rit_loaded() -> Any:
    """Include `jansen_rit.jl` into Julia Main on first use; return the module."""
    global _JANSEN_RIT_LOADED
    if _jl is None:
        raise ImportError("juliacall not available; install the `julia` extra")
    if not _JANSEN_RIT_LOADED:
        jl_path = _KERNEL_DIR / "jansen_rit.jl"
        if not jl_path.is_file():
            raise FileNotFoundError(f"jansen_rit.jl missing at {jl_path}")
        _jl.include(str(jl_path))
        _JANSEN_RIT_LOADED = True
    return _jl.JansenRitAccel


def _ensure_ermentrout_kopell_pop_loaded() -> Any:
    """Include the maintained MPR kernel on first use; return its module."""
    global _ERMENTROUT_KOPELL_POP_LOADED
    if _jl is None:
        raise ImportError("juliacall not available; install the `julia` extra")
    if not _ERMENTROUT_KOPELL_POP_LOADED:
        jl_path = _KERNEL_DIR / "ermentrout_kopell_pop.jl"
        if not jl_path.is_file():
            raise FileNotFoundError(f"ermentrout_kopell_pop.jl missing at {jl_path}")
        _jl.include(str(jl_path))
        _ERMENTROUT_KOPELL_POP_LOADED = True
    return _jl.ErmentroutKopellPopAccel


def _ensure_resonate_and_fire_loaded() -> Any:
    """Include the maintained resonate-and-fire kernel on first use."""
    global _RESONATE_AND_FIRE_LOADED
    if _jl is None:
        raise ImportError("juliacall not available; install the `julia` extra")
    if not _RESONATE_AND_FIRE_LOADED:
        jl_path = _KERNEL_DIR / "resonate_and_fire.jl"
        if not jl_path.is_file():
            raise FileNotFoundError(f"resonate_and_fire.jl missing at {jl_path}")
        _jl.include(str(jl_path))
        _RESONATE_AND_FIRE_LOADED = True
    return _jl.ResonateAndFireAccel


def _ensure_wilson_cowan_loaded() -> Any:
    """Include `wilson_cowan.jl` into Julia Main on first use; return the module."""
    global _WILSON_COWAN_LOADED
    if _jl is None:
        raise ImportError("juliacall not available; install the `julia` extra")
    if not _WILSON_COWAN_LOADED:
        jl_path = _KERNEL_DIR / "wilson_cowan.jl"
        if not jl_path.is_file():
            raise FileNotFoundError(f"wilson_cowan.jl missing at {jl_path}")
        _jl.include(str(jl_path))
        _WILSON_COWAN_LOADED = True
    return _jl.WilsonCowanAccel


def _as_wilson_cowan_ext_input(ext_input: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Convert the Wilson-Cowan drive into a one-dimensional float64 vector."""
    ext = np.ascontiguousarray(ext_input, dtype=np.float64)
    if ext.ndim != 1:
        raise ValueError(f"ext_input must be one-dimensional: got shape {ext.shape}")
    if not np.isfinite(ext).all():
        raise ValueError("ext_input must contain only finite values")
    return ext


def _as_wong_wang_inputs(
    stim1: npt.ArrayLike,
    stim2: npt.ArrayLike,
    xi: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Convert and validate Wong-Wang input traces for Julia dispatch."""
    stim1_arr = np.ascontiguousarray(stim1, dtype=np.float64)
    stim2_arr = np.ascontiguousarray(stim2, dtype=np.float64)
    xi_arr = np.ascontiguousarray(xi, dtype=np.float64)
    for name, array in (("stim1", stim1_arr), ("stim2", stim2_arr), ("xi", xi_arr)):
        if array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional: got shape {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")
    n = stim1_arr.size
    if stim2_arr.size != n:
        raise ValueError(f"stim1 and stim2 length mismatch: {n} vs {stim2_arr.size}")
    if xi_arr.size != 2 * n:
        raise ValueError(f"xi length must be 2 * n_steps ({2 * n}): got {xi_arr.size}")
    return stim1_arr, stim2_arr, xi_arr


def _as_jansen_rit_input(p_ext: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Convert the Jansen–Rit drive into a finite one-dimensional vector."""
    drive = np.ascontiguousarray(p_ext, dtype=np.float64)
    if drive.ndim != 1:
        raise ValueError(f"p_ext must be one-dimensional: got shape {drive.shape}")
    if not np.isfinite(drive).all():
        raise ValueError("p_ext must contain only finite values")
    return drive


def _as_ermentrout_kopell_pop_input(
    ext_input: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Convert the MPR drive into a finite one-dimensional vector."""
    drive = np.ascontiguousarray(ext_input, dtype=np.float64)
    if drive.ndim != 1:
        raise ValueError(f"ext_input must be one-dimensional: got shape {drive.shape}")
    if not np.isfinite(drive).all():
        raise ValueError("ext_input must contain only finite values")
    return drive


def _as_resonate_and_fire_input(
    current: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Convert resonate-and-fire drive into a finite one-dimensional vector."""
    drive = np.ascontiguousarray(current, dtype=np.float64)
    if drive.ndim != 1:
        raise ValueError(f"current must be one-dimensional: got shape {drive.shape}")
    if not np.isfinite(drive).all():
        raise ValueError("current must contain only finite values")
    return drive


def _ensure_rk4_neurons_loaded() -> Any:
    """Include `rk4_neurons.jl` into Julia Main on first use; return the module."""
    global _RK4_NEURONS_LOADED
    if _jl is None:
        raise ImportError("juliacall not available; install the `julia` extra")
    if not _RK4_NEURONS_LOADED:
        jl_path = _KERNEL_DIR / "rk4_neurons.jl"
        if not jl_path.is_file():
            raise FileNotFoundError(f"rk4_neurons.jl missing at {jl_path}")
        _jl.include(str(jl_path))
        _RK4_NEURONS_LOADED = True
    return _jl.Rk4NeuronsAccel


def _normalise_model_name(model_name: str) -> str:
    return "".join(ch.lower() for ch in model_name if ch.isalnum())


def simulate_wong_wang(
    s1_init: float,
    s2_init: float,
    noise1_init: float,
    noise2_init: float,
    tau_s: float,
    tau_ampa: float,
    gamma: float,
    j_n: float,
    j_cross: float,
    i_0: float,
    sigma: float,
    dt: float,
    stim1: npt.ArrayLike,
    stim2: npt.ArrayLike,
    xi: npt.ArrayLike,
) -> dict[str, npt.NDArray[np.float64] | float]:
    """Run the Julia implementation of the published Euler/OU recurrence.

    Parameters
    ----------
    s1_init, s2_init : float
        Initial NMDA gating fractions.
    noise1_init, noise2_init : float
        Initial AMPA Ornstein-Uhlenbeck current states.
    tau_s, tau_ampa, gamma, j_n, j_cross, i_0, sigma, dt : float
        Published reduced-model parameters.
    stim1, stim2 : ArrayLike
        Per-step external currents.
    xi : ArrayLike
        Interleaved standard-normal samples of length ``2 * n_steps``.

    Returns
    -------
    dict[str, numpy.ndarray | float]
        Six post-update traces and four final dynamic states.

    Raises
    ------
    ImportError
        If the Julia bridge is unavailable.
    FileNotFoundError
        If the maintained Wong-Wang kernel is absent.
    ValueError
        If an input stream violates the public shape or finite-value contract.
    """
    stim1_arr, stim2_arr, xi_arr = _as_wong_wang_inputs(stim1, stim2, xi)
    mod = _ensure_wong_wang_loaded()
    n = stim1_arr.size
    s1_out: npt.NDArray[np.float64] = np.empty(n, dtype=np.float64)
    s2_out: npt.NDArray[np.float64] = np.empty(n, dtype=np.float64)
    noise1_out: npt.NDArray[np.float64] = np.empty(n, dtype=np.float64)
    noise2_out: npt.NDArray[np.float64] = np.empty(n, dtype=np.float64)
    r1_out: npt.NDArray[np.float64] = np.empty(n, dtype=np.float64)
    r2_out: npt.NDArray[np.float64] = np.empty(n, dtype=np.float64)
    s1_final, s2_final, noise1_final, noise2_final = mod.simulate_wong_wang_b(
        s1_init,
        s2_init,
        noise1_init,
        noise2_init,
        tau_s,
        tau_ampa,
        gamma,
        j_n,
        j_cross,
        i_0,
        sigma,
        dt,
        stim1_arr,
        stim2_arr,
        xi_arr,
        s1_out,
        s2_out,
        noise1_out,
        noise2_out,
        r1_out,
        r2_out,
    )
    return {
        "s1": s1_out,
        "s2": s2_out,
        "noise1": noise1_out,
        "noise2": noise2_out,
        "r1": r1_out,
        "r2": r2_out,
        "s1_final": float(s1_final),
        "s2_final": float(s2_final),
        "noise1_final": float(noise1_final),
        "noise2_final": float(noise2_final),
    }


def simulate_jansen_rit(
    y0_init: float,
    y3_init: float,
    y1_init: float,
    y4_init: float,
    y2_init: float,
    y5_init: float,
    a_exc: float,
    b_exc: float,
    a_rate: float,
    b_rate: float,
    c: float,
    e0: float,
    v0: float,
    r: float,
    dt: float,
    p_ext: npt.ArrayLike,
) -> dict[str, npt.NDArray[np.float64] | float]:
    """Run the Julia equation-(6) recurrence and return complete traces."""
    drive = _as_jansen_rit_input(p_ext)
    module = _ensure_jansen_rit_loaded()
    traces: list[npt.NDArray[np.float64]] = [
        np.empty(drive.size, dtype=np.float64) for _ in range(7)
    ]
    finals = module.simulate_jansen_rit_b(
        y0_init,
        y3_init,
        y1_init,
        y4_init,
        y2_init,
        y5_init,
        a_exc,
        b_exc,
        a_rate,
        b_rate,
        c,
        e0,
        v0,
        r,
        dt,
        drive,
        *traces,
    )
    keys = ("y0", "y3", "y1", "y4", "y2", "y5", "eeg")
    result: dict[str, npt.NDArray[np.float64] | float] = {
        key: trace for key, trace in zip(keys, traces, strict=True)
    }
    for key, final in zip(keys[:6], finals, strict=True):
        result[f"{key}_final"] = float(final)
    return result


def simulate_ermentrout_kopell_pop(
    r_init: float,
    v_init: float,
    tau: float,
    delta: float,
    eta_bar: float,
    coupling: float,
    dt: float,
    ext_input: npt.ArrayLike,
) -> dict[str, npt.NDArray[np.float64] | float]:
    """Run the Julia implementation of the complete MPR recurrence.

    Parameters
    ----------
    r_init, v_init : float
        Initial population firing rate and mean membrane potential.
    tau, delta, eta_bar, coupling, dt : float
        Complete MPR configuration and explicit-Euler step.
    ext_input : ArrayLike
        One finite external drive value per step.

    Returns
    -------
    dict[str, numpy.ndarray | float]
        Post-update ``r`` and ``v`` traces plus both final-state receipts.

    Raises
    ------
    ImportError
        If the Julia bridge is unavailable.
    FileNotFoundError
        If the maintained MPR Julia kernel is absent.
    ValueError
        If the drive, configuration, or caller-owned buffers violate the
        maintained contract.
    FloatingPointError
        If a finite valid-entry recurrence produces an invalid candidate state.
    """
    drive = _as_ermentrout_kopell_pop_input(ext_input)
    module = _ensure_ermentrout_kopell_pop_loaded()
    r_out = np.empty(drive.size, dtype=np.float64)
    v_out = np.empty(drive.size, dtype=np.float64)
    try:
        r_final, v_final = module.simulate_ermentrout_kopell_pop_b(
            r_init,
            v_init,
            tau,
            delta,
            eta_bar,
            coupling,
            dt,
            drive,
            r_out,
            v_out,
        )
    except Exception as exc:
        if not is_julia_error(exc):
            raise
        julia_exception = getattr(exc, "exception", None)
        if module.is_configuration_error(julia_exception):
            raise ValueError(str(exc)) from exc
        if module.is_candidate_error(julia_exception):
            raise FloatingPointError(str(exc)) from exc
        raise
    return {
        "r": r_out,
        "v": v_out,
        "r_final": float(r_final),
        "v_final": float(v_final),
    }


def simulate_resonate_and_fire(
    x_init: float,
    y_init: float,
    b: float,
    omega: float,
    threshold: float,
    dt: float,
    current: npt.ArrayLike,
) -> dict[str, npt.NDArray[np.float64] | float | int]:
    """Run the Julia exact-flow recurrence with typed failure translation."""
    drive = _as_resonate_and_fire_input(current)
    module = _ensure_resonate_and_fire_loaded()
    x_out = np.empty(drive.size, dtype=np.float64)
    y_out = np.empty(drive.size, dtype=np.float64)
    spikes_out = np.empty(drive.size, dtype=np.float64)
    try:
        x_final, y_final, spike_count = module.simulate_resonate_and_fire_b(
            x_init,
            y_init,
            b,
            omega,
            threshold,
            dt,
            drive,
            x_out,
            y_out,
            spikes_out,
        )
    except Exception as exc:
        if not is_julia_error(exc):
            raise
        julia_exception = getattr(exc, "exception", None)
        if module.is_configuration_error(julia_exception):
            raise ValueError(str(exc)) from exc
        if module.is_candidate_error(julia_exception):
            raise FloatingPointError(str(exc)) from exc
        raise
    return {
        "x": x_out,
        "y": y_out,
        "spikes": spikes_out,
        "x_final": float(x_final),
        "y_final": float(y_final),
        "spike_count": int(spike_count),
    }


def simulate_wilson_cowan(
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
    ext_input: npt.ArrayLike,
) -> dict[str, npt.NDArray[np.float64] | float]:
    """Run the Julia-accelerated N-step Wilson-Cowan simulator.

    The external drive must be a one-dimensional time-series. The wrapper
    validates that shape before Julia dispatch so the kernel never receives an
    implicitly flattened matrix. Returned values match
    ``sc_neurocore_engine.py_wilson_cowan_simulate``: per-step ``e`` and ``i``
    arrays plus final scalar rates.
    """
    ext_input_arr = _as_wilson_cowan_ext_input(ext_input)
    mod = _ensure_wilson_cowan_loaded()
    n = ext_input_arr.size
    e_out: npt.NDArray[np.float64] = np.empty(n, dtype=np.float64)
    i_out: npt.NDArray[np.float64] = np.empty(n, dtype=np.float64)
    e_final, i_final = mod.simulate_wilson_cowan_b(
        e_init,
        i_init,
        w_ee,
        w_ei,
        w_ie,
        w_ii,
        tau_e,
        tau_i,
        a,
        theta,
        dt,
        ext_input_arr,
        e_out,
        i_out,
    )
    return {
        "e": e_out,
        "i": i_out,
        "e_final": float(e_final),
        "i_final": float(i_final),
    }


def simulate_rk4_neuron(
    model_name: str,
    current_trace: npt.ArrayLike,
    dt: float | None = None,
) -> dict[str, Any]:
    """Julia-backed RK4 batch simulator for the first priority neuron models.

    The returned schema mirrors ``sc_neurocore_engine.py_rk4_neuron_simulate``:
    state trajectories, zero-based spike indices, and ``n_steps``.
    """
    currents = cast(npt.NDArray[np.float64], np.asarray(current_trace, dtype=np.float64))
    if currents.ndim != 1:
        raise ValueError(f"current_trace must be 1-D, got {currents.ndim}-D")
    if currents.size == 0:
        raise ValueError("current_trace must be non-empty")
    if not np.isfinite(currents).all():
        raise ValueError("current_trace must contain only finite values")
    n = currents.size
    spikes: npt.NDArray[np.uint64] = np.empty(n, dtype=np.uint64)
    model = _normalise_model_name(model_name)
    mod = _ensure_rk4_neurons_loaded()

    if model in {"izhikevich", "scizhikevichneuron", "izhikevichneuron"}:
        izh_v: npt.NDArray[np.float64] = np.empty(n, dtype=np.float64)
        u: npt.NDArray[np.float64] = np.empty(n, dtype=np.float64)
        n_spikes = int(
            mod.simulate_izhikevich_rk4_b(currents, _dt_or_default(dt, 1.0), izh_v, u, spikes)
        )
        return {
            "v": izh_v,
            "u": u,
            "spikes": spikes[:n_spikes].copy(),
            "n_steps": n,
        }

    if model in {"hodgkinhuxley", "hodgkinhuxleyneuron"}:
        hh_v: npt.NDArray[np.float64] = np.empty(n, dtype=np.float64)
        m: npt.NDArray[np.float64] = np.empty(n, dtype=np.float64)
        h: npt.NDArray[np.float64] = np.empty(n, dtype=np.float64)
        gate_n: npt.NDArray[np.float64] = np.empty(n, dtype=np.float64)
        n_spikes = int(
            mod.simulate_hodgkin_huxley_rk4_b(
                currents,
                _dt_or_default(dt, 0.01),
                hh_v,
                m,
                h,
                gate_n,
                spikes,
            )
        )
        return {
            "v": hh_v,
            "m": m,
            "h": h,
            "n": gate_n,
            "spikes": spikes[:n_spikes].copy(),
            "n_steps": n,
        }

    if model in {"adex", "adexneuron"}:
        adex_v: npt.NDArray[np.float64] = np.empty(n, dtype=np.float64)
        w: npt.NDArray[np.float64] = np.empty(n, dtype=np.float64)
        n_spikes = int(
            mod.simulate_adex_rk4_b(currents, _dt_or_default(dt, 0.1), adex_v, w, spikes)
        )
        return {
            "v": adex_v,
            "w": w,
            "spikes": spikes[:n_spikes].copy(),
            "n_steps": n,
        }

    raise ValueError(f"unsupported RK4 neuron model {model_name!r}")


def _dt_or_default(dt: float | None, default: float) -> float:
    value = default if dt is None else dt
    if not isinstance(value, int | float) or not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError("dt must be a positive finite scalar")
    return float(value)


# Dedicated model facades import the shared Julia runtime above, so re-export
# them only after package initialisation is complete.
from .adaptive_threshold_if import (  # noqa: E402,F401
    _ensure_loaded as _ensure_adaptive_threshold_if_loaded,
)
from .adaptive_threshold_if import (  # noqa: E402,F401
    simulate_adaptive_threshold_if as simulate_adaptive_threshold_if,
)
