# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Neuron ODE templates for Studio Equation Playground

"""Curated ODE templates for the Studio equation playground."""

from __future__ import annotations

from typing import Any


TEMPLATES: dict[str, dict[str, Any]] = {
    "lif": {
        "name": "lif",
        "description": "Leaky Integrate-and-Fire",
        "equations": ["dv/dt = -(v - E_L) / tau_m + I / C"],
        "threshold": "v > -50",
        "reset": "v = -65",
        "params": {"E_L": -65.0, "tau_m": 10.0, "C": 1.0},
        "init": {"v": -65.0},
        "dt": 0.1,
        "current": 30.0,
        "duration": 100.0,
    },
    "izhikevich": {
        "name": "izhikevich",
        "description": "Izhikevich (regular spiking)",
        "equations": [
            "dv/dt = 0.04 * v**2 + 5 * v + 140 - u + I",
            "du/dt = a * (b * v - u)",
        ],
        "threshold": "v > 30",
        "reset": "v = c; u = u + d",
        "params": {"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0},
        "init": {"v": -65.0, "u": -14.0},
        "dt": 0.1,
        "current": 10.0,
        "duration": 200.0,
    },
    "adex": {
        "name": "adex",
        "description": "Adaptive Exponential Integrate-and-Fire",
        "equations": [
            "dv/dt = (-(v - E_L) + delta_T * exp((v - v_T) / delta_T) - w + I) / tau_m",
            "dw/dt = (a * (v - E_L) - w) / tau_w",
        ],
        "threshold": "v > 0",
        "reset": "v = -68; w = w + b",
        "params": {
            "E_L": -70.6,
            "v_T": -50.4,
            "delta_T": 2.0,
            "tau_m": 9.4,
            "tau_w": 144.0,
            "a": 0.004,
            "b": 0.0805,
        },
        "init": {"v": -70.6, "w": 0.0},
        "dt": 0.1,
        "current": 5.0,
        "duration": 200.0,
    },
    "hodgkin_huxley": {
        "name": "hodgkin_huxley",
        "description": "Hodgkin-Huxley (4-variable conductance model)",
        "equations": [
            "dv/dt = (-g_L * (v - E_L) - g_Na * m**3 * h * (v - E_Na) - g_K * n**4 * (v - E_K) + I) / C",
            "dm/dt = 0.1 * (v + 40) / (1 - exp(-(v + 40) / 10)) * (1 - m) - 4 * exp(-(v + 65) / 18) * m",
            "dh/dt = 0.07 * exp(-(v + 65) / 20) * (1 - h) - 1 / (1 + exp(-(v + 35) / 10)) * h",
            "dn/dt = 0.01 * (v + 55) / (1 - exp(-(v + 55) / 10)) * (1 - n) - 0.125 * exp(-(v + 65) / 80) * n",
        ],
        "threshold": "v > 0",
        "reset": "",
        "params": {
            "C": 1.0,
            "g_L": 0.3,
            "g_Na": 120.0,
            "g_K": 36.0,
            "E_L": -54.4,
            "E_Na": 50.0,
            "E_K": -77.0,
        },
        "init": {"v": -65.0, "m": 0.05, "h": 0.6, "n": 0.32},
        "dt": 0.01,
        "current": 10.0,
        "duration": 50.0,
    },
    "fitzhugh_nagumo": {
        "name": "fitzhugh_nagumo",
        "description": "FitzHugh-Nagumo (2-variable oscillator)",
        "equations": [
            "dv/dt = v - v**3 / 3 - w + I",
            "dw/dt = epsilon * (v + a - b * w)",
        ],
        "threshold": "v > 1.0",
        "reset": "",
        "params": {"epsilon": 0.08, "a": 0.7, "b": 0.8},
        "init": {"v": -1.0, "w": -0.5},
        "dt": 0.1,
        "current": 0.5,
        "duration": 300.0,
    },
}


def list_templates() -> list[dict[str, Any]]:
    """Return all curated Studio equation templates."""
    return list(TEMPLATES.values())


def get_template(name: str) -> dict[str, Any] | None:
    """Return one curated Studio equation template by name."""
    return TEMPLATES.get(name)
