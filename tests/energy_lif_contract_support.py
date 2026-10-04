# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — configured EnergyLIF contract profiles

"""Define source-model configurations shared by real EnergyLIF boundary tests."""

from __future__ import annotations

PARAMETERS: tuple[tuple[str, float, float], ...] = (
    ("v", -61.0, -60.5),
    ("epsilon", 0.32, 0.35),
    ("capacitance", 100.0, 110.0),
    ("g_leak", 9.0, 8.5),
    ("e_0", -62.5, -63.0),
    ("e_u", -58.5, -58.0),
    ("e_d", -40.0, -42.0),
    ("e_f", -62.0, -63.0),
    ("v_threshold", -59.0, -58.8),
    ("v_reset", -62.0, -62.2),
    ("alpha", 1.0, 0.9),
    ("epsilon_0", 0.5, 0.55),
    ("epsilon_c", 0.18, 0.20),
    ("delta", 0.01, 0.012),
    ("tau_e", 200.0, 220.0),
    ("dt", 0.1, 0.05),
)
CONFIGURATIONS: tuple[dict[str, float], ...] = (
    {},
    *({name: value} for name, _, value in PARAMETERS),
    {name: value for name, _, value in PARAMETERS},
    {"v": -58.8, "epsilon": 0.32},
    {"v": -58.8, "epsilon": 0.17},
)
INVALID_CONFIGURATIONS: tuple[dict[str, float], ...] = (
    {"epsilon_0": 0.0},
    {"epsilon_0": -0.1},
    {"e_0": -201.0},
    {"e_0": 101.0},
    {"alpha": 11.0},
    {"alpha": 1.0e308, "epsilon_0": 2.0},
    {"alpha": 1.0e-300, "epsilon_0": 1.0e-300},
    {"v": -201.0},
    {"v": 101.0},
    {"v_reset": -201.0},
    {"v_reset": 101.0},
    {"epsilon": -0.1},
    {"epsilon": 5.1},
    {"capacitance": 0.0},
    {"g_leak": 0.0},
    {"alpha": 0.0},
    {"tau_e": 0.0},
    {"dt": 0.0},
    {"dt": 1.1},
    {"dt": 0.2, "tau_e": 0.1},
    {"epsilon_c": -0.1},
    {"delta": -0.1},
    {"e_d": -62.0},
    {"v_threshold": -62.0},
)


def configuration_values(parameters: dict[str, float]) -> tuple[float, ...]:
    """Return all sixteen parameters in the actual native ABI order."""
    return tuple(parameters.get(name, default) for name, default, _ in PARAMETERS)
