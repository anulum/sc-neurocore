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

_BENDA_HERZ_LOADED = False
_SC_STOCHASTIC_RATE_ADAPTATION_LOADED = False

def _ensure_benda_herz_loaded() -> Any:
    """Include the source-bound Benda-Herz module."""
    global _BENDA_HERZ_LOADED
    if _jl is None:
        raise ImportError("juliacall not available; install the `julia` extra")
    if not _BENDA_HERZ_LOADED:
        jl_path = _KERNEL_DIR / "benda_herz.jl"
        if not jl_path.is_file():
            raise FileNotFoundError(f"benda_herz.jl missing at {jl_path}")
        _jl.include(str(jl_path))
        _BENDA_HERZ_LOADED = True
    return _jl.BendaHerzAccel


def simulate_benda_herz(
    currents: npt.ArrayLike,
    *,
    a: float = 0.0,
    phase: float = 0.0,
    onset_gain: float = 60.0,
    rheobase: float = 0.0,
    adaptation_slope: float = 0.1,
    tau_a: float = 100.0,
    dt: float = 0.1,
) -> dict[str, object]:
    """Run a source Benda-Herz trace in Julia."""
    drive = np.ascontiguousarray(currents, dtype=np.float64)
    module = _ensure_benda_herz_loaded()
    state = module.BendaHerzNeuronState(a, phase, onset_gain, rheobase, adaptation_slope, tau_a, dt)
    result = module.simulate(drive, state=state)
    return {
        "adaptation": np.asarray(result.adaptation, dtype=np.float64),
        "phases": np.asarray(result.phases, dtype=np.float64),
        "events": np.asarray(result.events, dtype=np.int64),
        "a_final": float(result.state.a),
        "phase_final": float(result.state.phase),
    }


def _ensure_sc_stochastic_rate_adaptation_loaded() -> Any:
    """Include the retained SC stochastic adaptation module."""
    global _SC_STOCHASTIC_RATE_ADAPTATION_LOADED
    if _jl is None:
        raise ImportError("juliacall not available; install the `julia` extra")
    if not _SC_STOCHASTIC_RATE_ADAPTATION_LOADED:
        jl_path = _KERNEL_DIR / "sc_stochastic_rate_adaptation.jl"
        if not jl_path.is_file():
            raise FileNotFoundError(f"sc_stochastic_rate_adaptation.jl missing at {jl_path}")
        _jl.include(str(jl_path))
        _SC_STOCHASTIC_RATE_ADAPTATION_LOADED = True
    return _jl.SCStochasticRateAdaptationAccel


def simulate_sc_stochastic_rate_adaptation(
    currents: npt.ArrayLike,
    uniforms: npt.ArrayLike,
    *,
    a: float = 0.0,
    f_max: float = 200.0,
    beta: float = 0.1,
    i_half: float = 5.0,
    tau_a: float = 100.0,
    delta_a: float = 0.5,
    dt: float = 1.0,
) -> dict[str, object]:
    """Run the retained SC recurrence with controlled uniforms in Julia."""
    drive = np.ascontiguousarray(currents, dtype=np.float64)
    randoms = np.ascontiguousarray(uniforms, dtype=np.float64)
    module = _ensure_sc_stochastic_rate_adaptation_loaded()
    state = module.SCStochasticRateAdaptationNeuronState(
        a, f_max, beta, i_half, tau_a, delta_a, dt, 0.0
    )
    result = module.simulate_controlled(drive, randoms, state=state)
    return {
        "adaptation": np.asarray(result.adaptation, dtype=np.float64),
        "events": np.asarray(result.events, dtype=np.int64),
        "a_final": float(result.state.a),
    }
