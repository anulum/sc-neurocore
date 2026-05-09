# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo-backed RK4 neuron integrators (ctypes dispatch)

"""Python entry point for the Mojo-compiled RK4 neuron simulators.

Build:

    cd src/sc_neurocore/accel/mojo/rk4_neurons
    ~/.pixi/bin/mojo build --emit shared-lib -o librk4_neurons.so rk4_neurons.mojo

The shared object is platform-specific and gitignored; the Mojo source is
tracked. The returned schema mirrors the Rust, Julia, and Go RK4 dispatchers.
"""

from __future__ import annotations

import ctypes
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

_LIB_PATH = Path(__file__).resolve().parent / "librk4_neurons.so"
_lib: ctypes.CDLL | None

try:
    _lib = ctypes.CDLL(str(_LIB_PATH))
    _lib.simulate_izhikevich_rk4_c.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    _lib.simulate_izhikevich_rk4_c.restype = ctypes.c_int
    _lib.simulate_adex_rk4_c.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    _lib.simulate_adex_rk4_c.restype = ctypes.c_int
    _lib.simulate_hodgkin_huxley_rk4_c.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    _lib.simulate_hodgkin_huxley_rk4_c.restype = ctypes.c_int
    _HAS_MOJO_RK4_NEURONS = True
except OSError:
    _lib = None
    _HAS_MOJO_RK4_NEURONS = False


def _normalise_model_name(model_name: str) -> str:
    return "".join(ch.lower() for ch in model_name if ch.isalnum())


def _current_trace(current_trace: npt.ArrayLike) -> npt.NDArray[np.float64]:
    currents = np.ascontiguousarray(current_trace, dtype=np.float64)
    if currents.ndim != 1:
        raise ValueError(f"current_trace must be 1-D, got {currents.ndim}-D")
    if currents.size == 0:
        raise ValueError("current_trace must be non-empty")
    if not np.isfinite(currents).all():
        raise ValueError("current_trace must contain only finite values")
    return cast(npt.NDArray[np.float64], currents)


def _dt_or_default(dt: float | None, default: float) -> float:
    value = default if dt is None else dt
    if not isinstance(value, int | float) or not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError("dt must be a positive finite scalar")
    return float(value)


def _missing_library_error() -> ImportError:
    return ImportError(
        f"librk4_neurons.so not built. Run: cd {_LIB_PATH.parent} && "
        f"~/.pixi/bin/mojo build --emit shared-lib -o {_LIB_PATH.name} rk4_neurons.mojo"
    )


def simulate_rk4_neuron(
    model_name: str,
    current_trace: npt.ArrayLike,
    dt: float | None = None,
) -> dict[str, Any]:
    """Mojo-backed RK4 batch simulator for priority neuron models."""
    if _lib is None:
        raise _missing_library_error()

    currents = _current_trace(current_trace)
    n = currents.size
    spikes: npt.NDArray[np.uint64] = np.empty(n, dtype=np.uint64)
    model = _normalise_model_name(model_name)

    if model in {"izhikevich", "scizhikevichneuron", "izhikevichneuron"}:
        izh_v: npt.NDArray[np.float64] = np.empty(n, dtype=np.float64)
        u: npt.NDArray[np.float64] = np.empty(n, dtype=np.float64)
        n_spikes = _lib.simulate_izhikevich_rk4_c(
            ctypes.c_int(n),
            ctypes.c_double(_dt_or_default(dt, 1.0)),
            currents.ctypes.data,
            izh_v.ctypes.data,
            u.ctypes.data,
            spikes.ctypes.data,
        )
        return {"v": izh_v, "u": u, "spikes": spikes[:n_spikes].copy(), "n_steps": n}

    if model in {"hodgkinhuxley", "hodgkinhuxleyneuron"}:
        hh_v: npt.NDArray[np.float64] = np.empty(n, dtype=np.float64)
        m: npt.NDArray[np.float64] = np.empty(n, dtype=np.float64)
        h: npt.NDArray[np.float64] = np.empty(n, dtype=np.float64)
        gate_n: npt.NDArray[np.float64] = np.empty(n, dtype=np.float64)
        n_spikes = _lib.simulate_hodgkin_huxley_rk4_c(
            ctypes.c_int(n),
            ctypes.c_double(_dt_or_default(dt, 0.01)),
            currents.ctypes.data,
            hh_v.ctypes.data,
            m.ctypes.data,
            h.ctypes.data,
            gate_n.ctypes.data,
            spikes.ctypes.data,
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
        n_spikes = _lib.simulate_adex_rk4_c(
            ctypes.c_int(n),
            ctypes.c_double(_dt_or_default(dt, 0.1)),
            currents.ctypes.data,
            adex_v.ctypes.data,
            w.ctypes.data,
            spikes.ctypes.data,
        )
        return {"v": adex_v, "w": w, "spikes": spikes[:n_spikes].copy(), "n_steps": n}

    raise ValueError(f"unsupported RK4 neuron model {model_name!r}")
