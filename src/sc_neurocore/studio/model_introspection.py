# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared Studio model introspection helpers

"""Class loading and field classification shared by catalogue and simulation."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import math
from typing import Any

from sc_neurocore.neurons.models import _CLASS_TO_MODULE

# State variable names that change during .step() — common across models
_KNOWN_STATE_VARS = {
    "v",
    "m",
    "h",
    "n",
    "w",
    "u",
    "g_e",
    "g_i",
    "s",
    "r",
    "q",
    "ca",
    "ca_i",
    "ca_concentration",
    "a",
    "b",
    "z",
    "x",
    "y",
    "phase",
    "amplitude",
    "trace",
    "s_trace",
    "refractory_timer",
    "n_k",
    "h_na",
    "m_na",
    "m_t",
    "h_t",
    "m_a",
    "h_a",
    "m_kd",
    "m_h",
    "m_ca",
    "h_ca",
    "m_nap",
    "h_nap",
}


_class_cache: dict[str, type] = {}


def _load_class(name: str) -> type:
    if name in _class_cache:
        return _class_cache[name]
    module_name = _CLASS_TO_MODULE[name]
    mod = importlib.import_module(f"sc_neurocore.neurons.models.{module_name}")
    cls: type = getattr(mod, name)
    _class_cache[name] = cls
    return cls


def _model_field_specs(cls: type) -> list[tuple[str, float]]:
    """Return ``(name, numeric-default)`` specs for a model class.

    Works for dataclass models (declared fields) and plain classes (the numeric
    keyword parameters of ``__init__``) so the catalogue can browse any
    registered model, not only dataclasses. Non-numeric parameters (identifiers,
    pools, flags) are skipped for plain classes; missing or non-numeric dataclass
    defaults are reported as ``0.0`` to preserve the historical contract.
    """
    if dataclasses.is_dataclass(cls):
        specs: list[tuple[str, float]] = []
        for f in dataclasses.fields(cls):
            default = f.default if f.default is not dataclasses.MISSING else 0.0
            value = (
                float(default)
                if isinstance(default, (int, float)) and not isinstance(default, bool)
                else 0.0
            )
            specs.append((f.name, value))
        return specs
    try:
        signature = inspect.signature(cls)
    except (TypeError, ValueError):
        return []
    plain: list[tuple[str, float]] = []
    for pname, param in signature.parameters.items():
        if pname == "self" or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        default = param.default
        if isinstance(default, bool) or not isinstance(default, (int, float)):
            continue
        plain.append((pname, float(default)))
    return plain


def _fixed_step_attribute(cls: type) -> float | None:
    """Return a positive numeric class-level ``dt`` that is not a constructor field.

    Integer and map models may fix their step as a class attribute (for example
    ``IntegerQIFNeuron.dt = 1.0``). Such a step cannot be overridden through the
    constructor but is the model's real time base, so the catalogue and the run
    contract must report it instead of the Studio default.
    """
    attribute = inspect.getattr_static(cls, "dt", None)
    if isinstance(attribute, bool) or not isinstance(attribute, (int, float)):
        return None
    value = float(attribute)
    if not math.isfinite(value) or value <= 0.0:
        return None
    return value


def _extract_dt(cls: type) -> float:
    """Return the model's default timestep, or ``0.1`` when undeclared.

    Resolution order: declared ``dt`` field default, fixed class-level ``dt``
    attribute, Studio default of ``0.1`` ms per step.
    """
    for name, default in _model_field_specs(cls):
        if name == "dt":
            return default
    fixed = _fixed_step_attribute(cls)
    if fixed is not None:
        return fixed
    return 0.1


def _classify_fields(cls: type) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split model fields into state variables and parameters."""
    state_vars: list[dict[str, Any]] = []
    params: list[dict[str, Any]] = []
    for name, default in _model_field_specs(cls):
        if name == "dt":
            continue
        entry = {"name": name, "default": default}
        if name in _KNOWN_STATE_VARS or name.startswith("v") and len(name) <= 2:
            state_vars.append(entry)
        elif name.startswith(
            ("v_", "e_", "g_", "tau_", "c_", "sigma", "alpha", "beta")
        ) or name.endswith(("_threshold", "_reset", "_rest", "_rev", "_max", "_min")):
            params.append(entry)
        elif name in _KNOWN_STATE_VARS:
            state_vars.append(entry)
        else:
            params.append(entry)
    if not state_vars:
        state_vars.append({"name": "v", "default": -65.0})
    return state_vars, params


_CATEGORY_RULES = [
    (
        "Conductance",
        [
            "HodgkinHuxley",
            "ConnorStevens",
            "WangBuzsaki",
            "TraubMiles",
            "PinskyRinzel",
            "MainenSejnowski",
            "BoothRinzel",
            "HayL5",
            "COBA",
            "TwoCompartment",
            "ReducedTraub",
        ],
    ),
    (
        "Integrate-and-Fire",
        [
            "LIF",
            "IF",
            "QIF",
            "EIF",
            "AdEx",
            "CLIF",
            "Adaptive",
            "GIF",
            "GLIF",
            "Mihalas",
            "Brette",
            "Integer",
        ],
    ),
    (
        "Oscillator",
        [
            "FitzHugh",
            "MorrisLecar",
            "Hindmarsh",
            "VanDerPol",
            "Theta",
            "Selkov",
            "Oregonator",
            "Lotka",
        ],
    ),
    ("Bursting", ["Chay", "Izhikevich", "Bertram", "Butera", "Rulkov", "Map"]),
    ("Hardware", ["Loihi", "SpiNNaker", "Akida", "BrainScale", "TrueNorth", "DPI", "Xylo"]),
    (
        "Network/Population",
        [
            "WilsonCowan",
            "WongWang",
            "JansenRit",
            "Wendling",
            "Ermentrout",
            "Amari",
            "Compte",
            "Larter",
        ],
    ),
    (
        "Statistical",
        [
            "Poisson",
            "Gamma",
            "GLM",
            "SpikeResponse",
            "GalvesLocherbach",
            "McCullochPitts",
            "Renewal",
        ],
    ),
    ("AI-Optimized", ["Attention", "Compositional", "CFC", "Arcane"]),
]


def _categorize(name: str) -> str:
    for category, keywords in _CATEGORY_RULES:
        if any(kw in name for kw in keywords):
            return category
    return "Other"
