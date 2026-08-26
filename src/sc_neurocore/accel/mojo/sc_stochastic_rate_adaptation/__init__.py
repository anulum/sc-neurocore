# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

"""Load the executable Mojo SC stochastic rate-adaptation ABI."""

from __future__ import annotations
import ctypes
from pathlib import Path
from typing import Any
import numpy as np
import numpy.typing as npt

_fn: Any | None
try:
    _lib = ctypes.CDLL(str(Path(__file__).with_name("libsc_stochastic_rate_adaptation.so")))
    _fn = _lib.sc_sra_simulate_c
    _fn.restype = ctypes.c_int
    _HAS_MOJO_SC_STOCHASTIC_RATE_ADAPTATION = True
except (OSError, AttributeError):
    _fn = None
    _HAS_MOJO_SC_STOCHASTIC_RATE_ADAPTATION = False


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
    if _fn is None:
        raise RuntimeError("Mojo SC stochastic rate-adaptation library unavailable")
    drive = np.ascontiguousarray(currents, dtype=np.float64)
    randoms = np.ascontiguousarray(uniforms, dtype=np.float64)
    adaptation = np.empty(drive.size)
    events = np.empty(drive.size, dtype=np.int64)
    af = np.empty(1)
    config = tuple(ctypes.c_double(x) for x in (a, f_max, beta, i_half, tau_a, delta_a, dt))
    arrays = (drive, randoms, adaptation, events, af)
    addresses = tuple(ctypes.c_ssize_t(values.ctypes.data) for values in arrays)
    code = _fn(ctypes.c_ssize_t(drive.size), *config, *addresses)
    if code:
        raise FloatingPointError(f"Mojo SC stochastic batch failed with status {code}")
    return {"adaptation": adaptation, "events": events, "a_final": float(af[0])}
