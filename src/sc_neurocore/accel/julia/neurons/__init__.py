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
    from juliacall import Main as _jl

    _HAS_JULIA_NEURONS = True
except ImportError:
    _jl = None
    _HAS_JULIA_NEURONS = False


_KERNEL_DIR = Path(__file__).resolve().parent
_WONG_WANG_LOADED = False
_WILSON_COWAN_LOADED = False
_RK4_NEURONS_LOADED = False


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
    stim1_arr = np.asarray(stim1, dtype=np.float64)
    stim2_arr = np.asarray(stim2, dtype=np.float64)
    xi_arr = np.asarray(xi, dtype=np.float64)
    for name, array in (("stim1", stim1_arr), ("stim2", stim2_arr), ("xi", xi_arr)):
        if array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional: got shape {array.shape}")
    n = stim1_arr.size
    if stim2_arr.size != n:
        raise ValueError(f"stim1 and stim2 length mismatch: {n} vs {stim2_arr.size}")
    if xi_arr.size != 2 * n:
        raise ValueError(f"xi length must be 2 * n_steps ({2 * n}): got {xi_arr.size}")
    return stim1_arr, stim2_arr, xi_arr


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
    tau_s: float,
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
    """Run the Julia-accelerated N-step Wong-Wang simulator.

    The stimulus and noise traces must be one-dimensional time-series. The
    wrapper validates their shapes before Julia dispatch so the kernel never
    receives implicitly flattened matrices. Returned values match
    ``sc_neurocore_engine.py_wong_wang_simulate``: per-step ``s1``, ``s2``,
    ``r1``, and ``r2`` arrays plus final scalar states.
    """
    stim1_arr, stim2_arr, xi_arr = _as_wong_wang_inputs(stim1, stim2, xi)
    mod = _ensure_wong_wang_loaded()
    n = stim1_arr.size
    s1_out: npt.NDArray[np.float64] = np.empty(n, dtype=np.float64)
    s2_out: npt.NDArray[np.float64] = np.empty(n, dtype=np.float64)
    r1_out: npt.NDArray[np.float64] = np.empty(n, dtype=np.float64)
    r2_out: npt.NDArray[np.float64] = np.empty(n, dtype=np.float64)
    s1_final, s2_final = mod.simulate_wong_wang_b(
        s1_init,
        s2_init,
        tau_s,
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
        r1_out,
        r2_out,
    )
    return {
        "s1": s1_out,
        "s2": s2_out,
        "r1": r1_out,
        "r2": r2_out,
        "s1_final": float(s1_final),
        "s2_final": float(s2_final),
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
