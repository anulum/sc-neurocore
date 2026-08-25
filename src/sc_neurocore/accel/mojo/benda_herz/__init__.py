# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

"""Load the executable Mojo Benda-Herz ABI."""

from __future__ import annotations
import ctypes
from pathlib import Path
from typing import Any
import numpy as np
import numpy.typing as npt

_fn: Any | None
try:
    _lib = ctypes.CDLL(str(Path(__file__).with_name("libbenda_herz.so")))
    _fn = _lib.benda_herz_simulate_c
    _fn.restype = ctypes.c_int
    _HAS_MOJO_BENDA_HERZ = True
except (OSError, AttributeError):
    _fn = None
    _HAS_MOJO_BENDA_HERZ = False


def simulate_benda_herz(
    a: float,
    phase: float,
    onset_gain: float,
    rheobase: float,
    adaptation_slope: float,
    tau_a: float,
    dt: float,
    currents: npt.ArrayLike,
) -> dict[str, object]:
    if _fn is None:
        raise RuntimeError("Mojo Benda-Herz shared library unavailable")
    drive = np.ascontiguousarray(currents, dtype=np.float64)
    adaptation = np.empty(drive.size)
    phases = np.empty(drive.size)
    events = np.empty(drive.size, dtype=np.int64)
    af = np.empty(1)
    pf = np.empty(1)
    code = _fn(
        ctypes.c_ssize_t(drive.size),
        *(
            ctypes.c_double(x)
            for x in (a, phase, onset_gain, rheobase, adaptation_slope, tau_a, dt)
        ),
        ctypes.c_ssize_t(drive.ctypes.data),
        ctypes.c_ssize_t(adaptation.ctypes.data),
        ctypes.c_ssize_t(phases.ctypes.data),
        ctypes.c_ssize_t(events.ctypes.data),
        ctypes.c_ssize_t(af.ctypes.data),
        ctypes.c_ssize_t(pf.ctypes.data),
    )
    if code:
        raise FloatingPointError(f"Mojo Benda-Herz batch failed with status {code}")
    return {
        "adaptation": adaptation,
        "phases": phases,
        "events": events,
        "a_final": float(af[0]),
        "phase_final": float(pf[0]),
    }
