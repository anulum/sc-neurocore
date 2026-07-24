# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_wilson_cowan_mojo_parity.py

from __future__ import annotations

"""Parity between the Python primary and the Mojo N-step simulator.

Mojo's `exp` comes from libm; Rust's comes from std; both are
IEEE-compliant but last-ulp drift accumulates over thousands of
non-linear iterations. Tolerance set to 1e-9 empirically (measured
drift on ramp drive is ~1e-10)."""
import numpy as np
import pytest
from sc_neurocore.neurons.models.wilson_cowan import WilsonCowanUnit
from sc_neurocore.accel.mojo.wilson_cowan import (
    _HAS_MOJO_WILSON_COWAN,
    simulate_wilson_cowan,
)

if not _HAS_MOJO_WILSON_COWAN:
    pytest.skip(
        "libwilson_cowan.so not built (run mojo build in accel/mojo/wilson_cowan)",
        allow_module_level=True,
    )
DEFAULT_PARAMS = dict(
    w_ee=10.0,
    w_ei=6.0,
    w_ie=10.0,
    w_ii=1.0,
    tau_e=1.0,
    tau_i=2.0,
    a=1.2,
    theta=4.0,
    dt=0.1,
)


def _run_python(ext: np.ndarray):
    u = WilsonCowanUnit(**DEFAULT_PARAMS)
    e = np.empty(ext.size, dtype=np.float64)
    i = np.empty(ext.size, dtype=np.float64)
    for t in range(ext.size):
        u.step(float(ext[t]))
        e[t], i[t] = u.e, u.i
    return e, i


def _run_mojo(ext: np.ndarray):
    out = simulate_wilson_cowan(
        0.1,
        0.05,
        DEFAULT_PARAMS["w_ee"],
        DEFAULT_PARAMS["w_ei"],
        DEFAULT_PARAMS["w_ie"],
        DEFAULT_PARAMS["w_ii"],
        DEFAULT_PARAMS["tau_e"],
        DEFAULT_PARAMS["tau_i"],
        DEFAULT_PARAMS["a"],
        DEFAULT_PARAMS["theta"],
        DEFAULT_PARAMS["dt"],
        ext.astype(np.float64),
    )
    return out["e"], out["i"]


__all__ = [
    "np",
    "pytest",
    "WilsonCowanUnit",
    "_HAS_MOJO_WILSON_COWAN",
    "simulate_wilson_cowan",
    "DEFAULT_PARAMS",
    "_run_python",
    "_run_mojo",
]
