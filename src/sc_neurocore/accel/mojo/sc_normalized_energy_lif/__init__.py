# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Load the executable Mojo SC normalized EnergyLIF ABI."""

from __future__ import annotations
import ctypes
from pathlib import Path
from typing import Any, cast
import numpy as np

_LIBRARY = Path(__file__).with_name("libsc_normalized_energy_lif.so")
try:
    _loaded = ctypes.CDLL(str(_LIBRARY))
    _loaded_function: Any = _loaded.sc_normalized_energy_lif_simulate_c
    _loaded_function.restype = ctypes.c_int
    _function: Any | None = _loaded_function
    _HAS_MOJO_SC_NORMALIZED_ENERGY_LIF = True
except (OSError, AttributeError):
    _function = None
    _HAS_MOJO_SC_NORMALIZED_ENERGY_LIF = False


def simulate_sc_normalized_energy_lif(*args: object) -> dict[str, object]:
    """Run the complete configured Mojo SC normalized EnergyLIF trace."""
    if _function is None:
        raise RuntimeError("Mojo SC normalized EnergyLIF shared library is unavailable")
    config = tuple(float(cast(float, v)) for v in args[:11])
    currents = np.ascontiguousarray(args[11], dtype=np.float64)
    if currents.ndim != 1:
        raise ValueError("Mojo SC normalized EnergyLIF current must be one-dimensional")
    traces = [np.empty(currents.size, dtype=np.float64) for _ in range(2)]
    events = np.empty(currents.size, dtype=np.int64)
    finals = [np.empty(1, dtype=np.float64) for _ in range(2)]
    code = _function(
        ctypes.c_ssize_t(currents.size),
        *(ctypes.c_double(v) for v in config),
        ctypes.c_ssize_t(currents.ctypes.data),
        *(ctypes.c_ssize_t(v.ctypes.data) for v in traces),
        ctypes.c_ssize_t(events.ctypes.data),
        *(ctypes.c_ssize_t(v.ctypes.data) for v in finals),
    )
    if code != 0:
        raise FloatingPointError(f"Mojo SC normalized EnergyLIF batch failed with status {code}")
    return {
        "voltages": traces[0],
        "epsilon": traces[1],
        "events": events,
        "v_final": float(finals[0][0]),
        "epsilon_final": float(finals[1][0]),
    }


__all__ = ["_HAS_MOJO_SC_NORMALIZED_ENERGY_LIF", "simulate_sc_normalized_energy_lif"]
