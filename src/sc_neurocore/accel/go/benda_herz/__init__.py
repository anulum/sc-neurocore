# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

"""ctypes loader for the source Benda-Herz Go ABI."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import numpy.typing as npt


class _Out(ctypes.Structure):
    _fields_ = [
        ("a", ctypes.c_double),
        ("phase", ctypes.c_double),
        ("event", ctypes.c_int32),
        ("status", ctypes.c_int32),
    ]


try:
    library = ctypes.CDLL(str(Path(__file__).with_name("libbenda_herz.so")))
    library.benda_herz_step.argtypes = [ctypes.c_double] * 8
    library.benda_herz_step.restype = _Out
    _LIB: ctypes.CDLL | None = library
    _HAS_GO_BENDA_HERZ = True
except OSError:
    _LIB = None
    _HAS_GO_BENDA_HERZ = False


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
    if _LIB is None:
        raise RuntimeError("Go Benda-Herz shared library unavailable")
    drive = np.ascontiguousarray(currents, dtype=np.float64)
    adaptation = np.empty(drive.size)
    phases = np.empty(drive.size)
    events = np.empty(drive.size, dtype=np.int64)
    for index, current in enumerate(drive):
        out = _LIB.benda_herz_step(
            a, phase, onset_gain, rheobase, adaptation_slope, tau_a, dt, float(current)
        )
        if out.status:
            raise ValueError("Go Benda-Herz transition failed")
        a, phase = out.a, out.phase
        adaptation[index], phases[index], events[index] = a, phase, out.event
    return {
        "adaptation": adaptation,
        "phases": phases,
        "events": events,
        "a_final": a,
        "phase_final": phase,
    }
