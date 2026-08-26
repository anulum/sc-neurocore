# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

"""ctypes loader for the SC stochastic rate-adaptation Go ABI."""

from __future__ import annotations
import ctypes
from pathlib import Path
import numpy as np
import numpy.typing as npt


class _Out(ctypes.Structure):
    _fields_ = [("a", ctypes.c_double), ("event", ctypes.c_int32), ("status", ctypes.c_int32)]


try:
    library = ctypes.CDLL(str(Path(__file__).with_name("libsc_stochastic_rate_adaptation.so")))
    library.sc_sra_step.argtypes = [ctypes.c_double] * 9
    library.sc_sra_step.restype = _Out
    _LIB: ctypes.CDLL | None = library
    _HAS_GO_SC_STOCHASTIC_RATE_ADAPTATION = True
except OSError:
    _LIB = None
    _HAS_GO_SC_STOCHASTIC_RATE_ADAPTATION = False


def simulate_sc_stochastic_rate_adaptation(
    a: float,
    f_max: float,
    beta: float,
    i_half: float,
    tau_a: float,
    delta_a: float,
    dt: float,
    currents: npt.ArrayLike,
    uniforms: npt.ArrayLike,
) -> dict[str, object]:
    if _LIB is None:
        raise RuntimeError("Go SC stochastic rate-adaptation library unavailable")
    drive = np.ascontiguousarray(currents, dtype=np.float64)
    randoms = np.ascontiguousarray(uniforms, dtype=np.float64)
    adaptation = np.empty(drive.size)
    events = np.empty(drive.size, dtype=np.int64)
    for index, (current, uniform) in enumerate(zip(drive, randoms, strict=True)):
        transition = (
            a,
            f_max,
            beta,
            i_half,
            tau_a,
            delta_a,
            dt,
            float(current),
            float(uniform),
        )
        out = _LIB.sc_sra_step(*transition)
        if out.status:
            raise ValueError("Go SC stochastic transition failed")
        a = out.a
        adaptation[index] = a
        events[index] = out.event
    return {"adaptation": adaptation, "events": events, "a_final": a}
