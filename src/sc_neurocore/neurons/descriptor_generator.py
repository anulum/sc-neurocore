# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Model descriptor skeleton generator

"""Generate a v2 model descriptor skeleton from a registered model.

The skeleton carries everything that can be read honestly from the model itself
— constructor parameters with their defaults, state variables, the timestep, the
docstring summary, and any DOI the docstring already cites — plus any structured
fields a curated v1 ``model_schemas`` file already provides (provenance,
dynamics, declared state/parameter values). Curation-only fields (parameter
units, ranges, meanings, taxonomy, behaviour tags, the backend matrix beyond
Python, reproducibility) are left empty: the generator never fabricates a value.
The result is a curatable TOML/JSON payload that :func:`parse_model_descriptor`
validates.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import re
from collections.abc import Mapping
from typing import Any

from sc_neurocore.neurons.model_descriptor import (
    MODEL_DESCRIPTOR_SCHEMA_VERSION,
    ModelDescriptor,
    parse_model_descriptor,
)
from sc_neurocore.neurons.models import _CLASS_TO_MODULE

# Single letters that are conventionally state variables. ``a``/``b`` are
# intentionally excluded — across this library they are far more often
# parameters (adaptation increments, coupling constants) than state.
_KNOWN_STATE_VARS = frozenset({"v", "w", "u", "h", "n", "m", "ca", "s", "r", "theta", "vm"})
_PARAM_PREFIXES = ("v_", "e_", "g_", "tau_", "c_", "sigma", "alpha", "beta")
_PARAM_SUFFIXES = ("_threshold", "_reset", "_rest", "_rev", "_max", "_min")
_DOI_IN_TEXT = re.compile(r"10\.\d{4,9}/[^\s,)]+")


def _load_class(class_name: str) -> type:
    module = importlib.import_module(f"sc_neurocore.neurons.models.{_CLASS_TO_MODULE[class_name]}")
    return getattr(module, class_name)  # type: ignore[no-any-return]


def _field_specs(cls: type) -> list[tuple[str, float]]:
    """Return ``(name, numeric-default)`` for a dataclass or plain model class."""

    if dataclasses.is_dataclass(cls):
        specs: list[tuple[str, float]] = []
        for f in dataclasses.fields(cls):
            default = f.default
            # Skip non-numeric fields (string Literal integrator choices, flags,
            # labels): they are not numeric parameters of the model.
            if isinstance(default, bool) or not isinstance(default, (int, float)):
                continue
            specs.append((f.name, float(default)))
        return specs
    try:
        signature = inspect.signature(cls)
    except (TypeError, ValueError):
        return []
    plain: list[tuple[str, float]] = []
    for pname, param in signature.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        default = param.default
        if isinstance(default, bool) or not isinstance(default, (int, float)):
            continue
        plain.append((pname, float(default)))
    return plain


def _is_state_var(name: str) -> bool:
    return name in _KNOWN_STATE_VARS or (name.startswith("v") and len(name) <= 2)


def _is_param(name: str) -> bool:
    return name.startswith(_PARAM_PREFIXES) or name.endswith(_PARAM_SUFFIXES)


def _load_v1_schema(module: str) -> dict[str, Any]:
    """Return the curated v1 schema for a module, or an empty mapping."""

    try:
        from sc_neurocore.neurons.universal_dsl import load_schema

        return load_schema(module)
    except Exception:
        return {}


def _summary_from_docstring(cls: type) -> str:
    doc = (cls.__doc__ or "").strip()
    return doc.split("\n", 1)[0].strip() if doc else ""


def _doi_from_text(text: str) -> str:
    match = _DOI_IN_TEXT.search(text)
    return match.group(0).rstrip(".") if match else ""


def generate_descriptor_payload(class_name: str) -> dict[str, Any]:
    """Return a curatable v2 descriptor payload for a registered model.

    Parameters
    ----------
    class_name:
        Registered model class name (a key of the model registry).

    Returns
    -------
    dict[str, Any]
        A descriptor payload using the v2 section layout, with introspected and
        carried-over values filled and curation-only fields left empty.

    Raises
    ------
    KeyError
        If ``class_name`` is not registered.
    """

    if class_name not in _CLASS_TO_MODULE:
        raise KeyError(class_name)
    module = _CLASS_TO_MODULE[class_name]
    cls = _load_class(class_name)
    v1 = _load_v1_schema(module)
    v1_meta = v1.get("metadata", {}) if isinstance(v1.get("metadata"), Mapping) else {}
    v1_state = v1.get("state", {}) if isinstance(v1.get("state"), Mapping) else {}
    v1_params = v1.get("parameters", {}) if isinstance(v1.get("parameters"), Mapping) else {}

    specs = _field_specs(cls)
    state: dict[str, Any] = {}
    parameters: dict[str, Any] = {}
    dt = 0.1
    for name, default in specs:
        if name == "dt":
            dt = default
            continue
        if name in v1_state:
            state[name] = {"init": float(v1_state[name]) if _is_number(v1_state[name]) else default}
        elif name in v1_params or _is_param(name):
            parameters[name] = {"default": default, "unit": "", "meaning": ""}
        elif _is_state_var(name):
            state[name] = {"init": default}
        else:
            parameters[name] = {"default": default, "unit": "", "meaning": ""}
    if not state:
        state["v"] = {"init": -65.0}

    doc = cls.__doc__ or ""
    doi = ""
    raw_doi = v1_meta.get("doi")
    if isinstance(raw_doi, str) and raw_doi:
        doi = raw_doi
    else:
        doi = _doi_from_text(doc)
    year = v1_meta.get("year")
    authors: list[str] = []
    author_field = v1_meta.get("author")
    if isinstance(author_field, str) and author_field:
        authors = [author_field]

    provenance: dict[str, Any] = {}
    if authors:
        provenance["authors"] = authors
    if isinstance(year, int) and not isinstance(year, bool):
        provenance["year"] = year
    if doi:
        provenance["doi"] = doi

    dynamics: dict[str, Any] = {}
    v1_dyn = v1.get("dynamics")
    if isinstance(v1_dyn, Mapping):
        for var, expr in v1_dyn.items():
            if isinstance(expr, str):
                dynamics[var] = expr

    payload: dict[str, Any] = {
        "metadata": {
            "schema_version": MODEL_DESCRIPTOR_SCHEMA_VERSION,
            "name": str(v1_meta.get("name") or class_name),
            "class_name": class_name,
            "module": module,
            "display_name": "",
            "summary": str(v1_meta.get("description") or _summary_from_docstring(cls)),
            "family": "",
            "category": "",
            "biophysical_detail": "point",
            "maturity": "experimental",
            "intended_use": [],
            "hardware_fit": [],
            "behavior_tags": [],
        },
        "provenance": provenance,
        "state": state,
        "parameters": parameters,
        "integration": {
            "dt": dt,
            "method": str(v1.get("integration", {}).get("method", "euler"))
            if isinstance(v1.get("integration"), Mapping)
            else "euler",
        },
        "dynamics": dynamics,
        "backends": {"python": {"status": "implemented"}},
        "reproducibility": {},
        "documentation": {"slug": f"models/{module}", "notes": ""},
    }
    return payload


def generate_descriptor(class_name: str) -> ModelDescriptor:
    """Return a validated :class:`ModelDescriptor` skeleton for a model."""

    return parse_model_descriptor(generate_descriptor_payload(class_name))


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


__all__ = [
    "generate_descriptor",
    "generate_descriptor_payload",
]
