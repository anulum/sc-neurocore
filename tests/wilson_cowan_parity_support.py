# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_wilson_cowan_parity.py

from __future__ import annotations

"""Last-ulp parity between `WilsonCowanUnit.step` and the Rust
`py_wilson_cowan_simulate`.

Wilson-Cowan is deterministic (no stochastic noise), so Python and
Rust must produce numerically identical trajectories up to the
last few IEEE-754 ulps for any external-input sequence.
"""
import numpy as np
import pytest
from sc_neurocore.neurons.models.wilson_cowan import WilsonCowanUnit

pytest.importorskip(
    "sc_neurocore_engine", reason="Rust engine wheel not installed (maturin develop)"
)
from sc_neurocore_engine import py_wilson_cowan_simulate  # noqa: E402

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


def _run_rust(ext: np.ndarray):
    out = py_wilson_cowan_simulate(
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
    "py_wilson_cowan_simulate",
    "DEFAULT_PARAMS",
    "_run_python",
    "_run_rust",
]
