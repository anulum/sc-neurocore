# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — canonical NIR neuron ODE templates

"""Canonical ODE templates for the NIR neuron types SC-NeuroCore supports.

This is the single source of truth for the fixed-point neuron dynamics: the FPGA
compiler (:mod:`sc_neurocore.nir_bridge.fpga_compiler`) builds
:class:`~sc_neurocore.neurons.equation_builder.EquationNeuron` populations from
these templates, and the lightweight dict importer
(:mod:`sc_neurocore.compiler.intelligence.nir_import`) derives its per-node
equations from the same table, so the two paths cannot drift apart. The keys are
the internal neuron-type tags produced by
:mod:`sc_neurocore.nir_bridge.neuron_graph` (``lif``, ``if``, ``li``,
``cuba_lif``, ``cuba_li``, ``integrator``), plus two Studio catalogue profiles
(``sc_lif``, ``sc_if``) that only the Studio's network lowering selects. A
template may name its integration ``method`` (``euler`` when absent) and the
parameter its state starts from (``rest_param``, ``v_leak`` when absent). The module deliberately has no
third-party dependencies so either consumer can import it without pulling in the
heavier compilation stack.
"""

from __future__ import annotations

from typing import Any

NEURON_TEMPLATES: dict[str, dict[str, Any]] = {
    "lif": {
        "equations": ["dv/dt = -(v - v_leak) / tau + I * r / tau"],
        "threshold": "v > v_threshold",
        "reset": "v = v_reset",
        "default_params": {
            "tau": 20.0,
            "r": 1.0,
            "v_leak": 0.0,
            "v_threshold": 1.0,
            "v_reset": 0.0,
        },
    },
    "if": {
        "equations": ["dv/dt = I * r"],
        "threshold": "v > v_threshold",
        "reset": "v = v_reset",
        "default_params": {"r": 1.0, "v_threshold": 1.0, "v_reset": 0.0},
    },
    "li": {
        "equations": ["dv/dt = -(v - v_leak) / tau + I * r / tau"],
        "threshold": None,
        "reset": None,
        "default_params": {"tau": 20.0, "r": 1.0, "v_leak": 0.0},
    },
    "cuba_lif": {
        "equations": [
            "di_syn/dt = -i_syn / tau_syn + I * w_in",
            "dv/dt = -(v - v_leak) / tau_mem + i_syn * r / tau_mem",
        ],
        "threshold": "v > v_threshold",
        "reset": "v = v_reset",
        "default_params": {
            "tau_syn": 5.0,
            "tau_mem": 20.0,
            "r": 1.0,
            "v_leak": 0.0,
            "v_threshold": 1.0,
            "v_reset": 0.0,
            "w_in": 1.0,
        },
    },
    "cuba_li": {
        "equations": [
            "di_syn/dt = -i_syn / tau_syn + I * w_in",
            "dv/dt = -(v - v_leak) / tau_mem + i_syn * r / tau_mem",
        ],
        "threshold": None,
        "reset": None,
        "default_params": {"tau_syn": 5.0, "tau_mem": 20.0, "r": 1.0, "v_leak": 0.0, "w_in": 1.0},
    },
    "integrator": {
        "equations": ["dv/dt = I * r"],
        "threshold": None,
        "reset": None,
        "default_params": {"r": 1.0},
    },
    # Studio catalogue profiles. These are not NIR primitives and no NIR import
    # produces them: the Studio's network lowering selects them so the hardware
    # computes the catalogue model the Studio simulates, not NIR's version of it.
    # ``SCLapicqueLIFNeuron`` profile ``sc_lif`` integrates a step exactly,
    # v <- v_inf + (v - v_inf) * exp(-dt / tau) with v_inf = v_rest + R * I, so it
    # is a map with ``decay = exp(-dt / tau)`` and ``gain = 1 - decay``, and it
    # fires at v >= v_threshold.
    "sc_lif": {
        "equations": ["dv/dt = v * decay + (v_rest + I * r) * gain"],
        "method": "map",
        "rest_param": "v_rest",
        "threshold": "v >= v_threshold",
        "reset": "v = v_reset",
        "default_params": {
            "decay": 0.951229424500714,
            "gain": 0.048770575499285984,
            "v_rest": 0.0,
            "r": 1.0,
            "v_threshold": 1.0,
            "v_reset": 0.0,
        },
    },
    # ``PerfectIntegratorNeuron`` profile ``sc_inclusive``: the ``if`` dynamics
    # with the inclusive comparison the catalogue model uses.
    "sc_if": {
        "equations": ["dv/dt = I * r"],
        "threshold": "v >= v_threshold",
        "reset": "v = v_reset",
        "default_params": {"r": 1.0, "v_threshold": 1.0, "v_reset": 0.0},
    },
}

#: Templates the Studio's network lowering selects for catalogue models. They
#: are not NIR node types: no NIR import produces them, and the dict importer
#: refuses them (``sc_lif`` is a per-step map, not a derivative).
STUDIO_PROFILE_TEMPLATES: frozenset[str] = frozenset({"sc_lif", "sc_if"})

#: The NIR point-neuron types, every template that is not a Studio profile.
NIR_PRIMITIVE_TEMPLATES: frozenset[str] = frozenset(NEURON_TEMPLATES) - STUDIO_PROFILE_TEMPLATES
