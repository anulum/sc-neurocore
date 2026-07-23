# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_wilson_cowan.py

from __future__ import annotations

"""Full pipeline test for WilsonCowanUnit (Wilson & Cowan 1972).

E/I rate model: returns float (E rate), not int spike.
τ_e dE/dt = -E + S(w_ee·E - w_ei·I + I_ext).
Pipeline limited: returns float, Network expects int → documented.
Performance evidence is benchmark-only. CI checks bounded runtime under
coverage/load, not production throughput."""
import json
import math
from pathlib import Path
import time
try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib
import numpy as np
import pytest
from sc_neurocore.neurons.models.wilson_cowan import WilsonCowanUnit
from sc_neurocore.network.population import Population
_REPOSITORY = Path(__file__).resolve().parents[1]
def _rk4_expected_state(unit: WilsonCowanUnit, drive: float) -> tuple[float, float]:
    e0, i0 = unit.e, unit.i

    def derivatives(e: float, i: float) -> tuple[float, float]:
        se = unit._sigmoid(unit.w_ee * e - unit.w_ei * i + drive)
        si = unit._sigmoid(unit.w_ie * e - unit.w_ii * i)
        return (-e + se) / unit.tau_e, (-i + si) / unit.tau_i

    k1_e, k1_i = derivatives(e0, i0)
    k2_e, k2_i = derivatives(e0 + 0.5 * unit.dt * k1_e, i0 + 0.5 * unit.dt * k1_i)
    k3_e, k3_i = derivatives(e0 + 0.5 * unit.dt * k2_e, i0 + 0.5 * unit.dt * k2_i)
    k4_e, k4_i = derivatives(e0 + unit.dt * k3_e, i0 + unit.dt * k3_i)
    return (
        e0 + unit.dt * (k1_e + 2.0 * k2_e + 2.0 * k3_e + k4_e) / 6.0,
        i0 + unit.dt * (k1_i + 2.0 * k2_i + 2.0 * k3_i + k4_i) / 6.0,
    )

__all__ = ['json', 'math', 'Path', 'time', 'tomllib', 'np', 'pytest', 'WilsonCowanUnit', 'Population', '_REPOSITORY', '_rk4_expected_state']
