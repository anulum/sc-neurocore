# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — dedicated Julia neuron facade

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt

from ._runtime import JULIA_MAIN as _jl
from ._runtime import KERNEL_DIR as _KERNEL_DIR

_COMPTE_WM_LOADED = False

def _ensure_compte_wm_loaded() -> Any:
    """Include the native Compte module into Julia Main on first use."""
    global _COMPTE_WM_LOADED
    if _jl is None:
        raise ImportError("juliacall not available; install the julia extra")
    if not _COMPTE_WM_LOADED:
        jl_path = _KERNEL_DIR / "compte_wm.jl"
        if not jl_path.is_file():
            raise FileNotFoundError(f"compte_wm.jl missing at {jl_path}")
        _jl.include(str(jl_path))
        _COMPTE_WM_LOADED = True
    return _jl.CompteWmAccel


def simulate_compte_wm(*args: object) -> dict[str, object]:
    """Run complete Compte membrane, channel, refractory, and event traces."""
    config = tuple(float(cast(float, value)) for value in args[:24])
    inputs = (
        np.ascontiguousarray(cast(npt.ArrayLike, args[24]), dtype=np.float64),
        *(
            np.ascontiguousarray(cast(npt.ArrayLike, value), dtype=np.int64)
            for value in args[25:28]
        ),
    )
    steps = inputs[0].size
    keys = ("voltages", "s_ampa", "s_nmda", "x_nmda", "s_gaba", "refractory")
    final_keys = (
        "v_final",
        "s_ampa_final",
        "s_nmda_final",
        "x_nmda_final",
        "s_gaba_final",
        "ref_final",
    )
    traces = {key: np.empty(steps, dtype=np.float64) for key in keys}
    events = np.empty(steps, dtype=np.int64)
    module = _ensure_compte_wm_loaded()
    finals = module.simulate_compte_wm_b(
        *config,
        *inputs,
        *(traces[key] for key in keys),
        events,
    )
    return {
        **traces,
        "events": events,
        **{key: float(finals[index]) for index, key in enumerate(final_keys)},
    }
