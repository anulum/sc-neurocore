# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Load the native Go EnergyLIF batch."""

from __future__ import annotations
import ctypes
from pathlib import Path
from typing import Any, cast
import numpy as np

_LIBRARY = Path(__file__).with_name("libenergy_lif.so")
try:
    _loaded = ctypes.CDLL(str(_LIBRARY))
    _loaded_function: Any = _loaded.energy_lif_simulate_c
    _loaded_function.restype = ctypes.c_int
    _function: Any | None = _loaded_function
    _HAS_GO_ENERGY_LIF = True
except (OSError, AttributeError):
    _function = None
    _HAS_GO_ENERGY_LIF = False


def simulate_energy_lif(*args: object) -> dict[str, object]:
    """Run the complete configured Go EnergyLIF trace."""
    if _function is None:
        raise RuntimeError("Go EnergyLIF shared library is unavailable")
    config = tuple(float(cast(float, v)) for v in args[:16])
    currents = np.ascontiguousarray(args[16], dtype=np.float64)
    if currents.ndim != 1:
        raise ValueError("Go EnergyLIF current must be one-dimensional")
    traces = [np.empty(currents.size, dtype=np.float64) for _ in range(2)]
    events = np.empty(currents.size, dtype=np.int64)
    finals = [np.empty(1, dtype=np.float64) for _ in range(2)]
    code = _function(
        ctypes.c_int(currents.size),
        *(ctypes.c_double(v) for v in config),
        ctypes.c_void_p(currents.ctypes.data),
        *(ctypes.c_void_p(v.ctypes.data) for v in traces),
        ctypes.c_void_p(events.ctypes.data),
        *(ctypes.c_void_p(v.ctypes.data) for v in finals),
    )
    if code != 0:
        raise FloatingPointError(f"Go EnergyLIF batch failed with status {code}")
    return {
        "voltages": traces[0],
        "epsilon": traces[1],
        "events": events,
        "v_final": float(finals[0][0]),
        "epsilon_final": float(finals[1][0]),
    }


__all__ = ["_HAS_GO_ENERGY_LIF", "simulate_energy_lif"]
