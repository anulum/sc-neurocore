# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — dedicated Julia neuron facade

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from ._runtime import JULIA_MAIN as _jl
from ._runtime import KERNEL_DIR as _KERNEL_DIR

_BRUNEL_WANG_LOADED = False


def _ensure_brunel_wang_loaded() -> Any:
    """Include ``brunel_wang.jl`` into Julia Main on first use."""
    global _BRUNEL_WANG_LOADED
    if _jl is None:
        raise ImportError("juliacall not available; install the `julia` extra")
    if not _BRUNEL_WANG_LOADED:
        jl_path = _KERNEL_DIR / "brunel_wang.jl"
        if not jl_path.is_file():
            raise FileNotFoundError(f"brunel_wang.jl missing at {jl_path}")
        _jl.include(str(jl_path))
        _BRUNEL_WANG_LOADED = True
    return _jl.BrunelWangAccel


def simulate_brunel_wang(
    v: float,
    ref_remaining: float,
    v_rest: float,
    v_reset: float,
    v_threshold: float,
    tau_m: float,
    tau_ref: float,
    g_ampa_ext: float,
    g_ampa_rec: float,
    g_nmda: float,
    g_gaba: float,
    v_ampa: float,
    v_nmda: float,
    v_gaba: float,
    c_m: float,
    mg_conc: float,
    dt: float,
    ext: npt.NDArray[np.float64],
    ampa: npt.NDArray[np.float64],
    nmda: npt.NDArray[np.float64],
    gaba: npt.NDArray[np.float64],
) -> dict[str, object]:
    """Run complete voltage, refractory, and event traces in Julia."""
    steps = ext.size
    voltages = np.empty(steps, dtype=np.float64)
    refractory = np.empty(steps, dtype=np.float64)
    events = np.empty(steps, dtype=np.int64)
    module = _ensure_brunel_wang_loaded()
    finals = module.simulate_brunel_wang_b(
        v,
        ref_remaining,
        v_rest,
        v_reset,
        v_threshold,
        tau_m,
        tau_ref,
        g_ampa_ext,
        g_ampa_rec,
        g_nmda,
        g_gaba,
        v_ampa,
        v_nmda,
        v_gaba,
        c_m,
        mg_conc,
        dt,
        ext,
        ampa,
        nmda,
        gaba,
        voltages,
        refractory,
        events,
    )
    return {
        "voltages": voltages,
        "refractory": refractory,
        "events": events,
        "v_final": float(finals[0]),
        "ref_final": float(finals[1]),
    }
