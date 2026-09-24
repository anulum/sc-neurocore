# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — A real candidate package for the candidate tests

"""Build a valid candidate from a bundled schema, fresh for every test."""

from __future__ import annotations

import copy
from typing import Any

from sc_neurocore.neurons.universal_dsl import load_schema
from sc_neurocore.studio.candidate_package import CANDIDATE_SCHEMA_VERSION

_UNITS = {
    "v_rest": "mV",
    "v_reset": "mV",
    "v_threshold": "mV",
    "v_rh": "mV",
    "delta_T": "mV",
    "tau": "ms",
    "tau_w": "ms",
    "a": "nS",
    "b_adapt": "pA",
    "C": "pF",
}


def adex_candidate() -> dict[str, Any]:
    """Return a valid candidate: the bundled AdEx with a slower adaptation."""
    model = copy.deepcopy(load_schema("adex"))
    model["metadata"]["name"] = "SlowAdaptationAdEx"
    model["parameters"]["tau_w"] = 300.0
    return {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "name": "SlowAdaptationAdEx",
        "parent": "AdExNeuron",
        "model": model,
        "units": {
            "state": {"v": "mV", "w": "pA"},
            "parameters": dict(_UNITS),
            "current": "pA",
            "time": "ms",
        },
        "source": {
            "citation": "Brette, R. & Gerstner, W. (2005) J. Neurophysiol. 94:3637-3642",
            "doi": "10.1152/jn.00686.2005",
        },
        "assumptions": ["point neuron", "adaptation slowed to 300 ms for the study"],
        "authors": ["A. Researcher"],
        "reference_tests": [
            {
                "name": "rests without drive",
                "current": 0.0,
                "steps": 1000,
                "expect": {
                    "spike_count": {"max": 0},
                    "final_state": {"v": {"min": -66.0, "max": -64.0}},
                },
            },
            {
                "name": "fires under drive",
                "current": 800.0,
                "steps": 5000,
                "expect": {"spike_count": {"min": 1}},
            },
        ],
    }
