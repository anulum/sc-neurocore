# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Model descriptor skeleton generator

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

import ast
import dataclasses
import importlib
import inspect
import re
import textwrap
from collections.abc import Mapping
from typing import Any

from sc_neurocore.neurons.model_descriptor import (
    MODEL_DESCRIPTOR_SCHEMA_VERSION,
    ModelDescriptor,
    parse_model_descriptor,
)
from sc_neurocore.neurons.model_taxonomy import model_family
from sc_neurocore.neurons.models import _CLASS_TO_MODULE

# Single letters that are conventionally state variables. ``a``/``b`` are
# intentionally excluded — across this library they are far more often
# parameters (adaptation increments, coupling constants) than state. ``x``/``y``/
# ``z`` are the integration coordinates of the map-based, resonate-and-fire, and
# three-variable models (suffixed forms like ``x_rest`` stay parameters via
# ``_is_param``).
_KNOWN_STATE_VARS = frozenset(
    {
        "v",
        "w",
        "u",
        "h",
        "n",
        "m",
        "ca",
        "s",
        "s1",
        "s2",
        "q",
        "r",
        "theta",
        "vm",
        "x",
        "y",
        "z",
        "time_since_spike",
    }
)
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
            # Skip private implementation fields (integration sub-step counts,
            # clamp constants, RNG and history buffers): they are not model
            # parameters or declared state.
            if f.name.startswith("_"):
                continue
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
        if pname.startswith("_"):
            continue
        default = param.default
        if isinstance(default, bool) or not isinstance(default, (int, float)):
            continue
        plain.append((pname, float(default)))
    return plain


def _is_state_var(name: str) -> bool:
    # Exact membership, plus ion-channel gating variables: ``m_``/``h_``-prefixed
    # names (m_na, h_cat, h_nap, ...) are activation/inactivation gates, which are
    # integration state. Suffixed parameters (h_max, m_min) are caught by
    # ``_is_param`` first, which takes precedence in the classifier. A ``v``-prefixed
    # two-letter name (v0, v1) is a reversal/midpoint parameter, not state.
    return name in _KNOWN_STATE_VARS or name.startswith(("m_", "h_"))


def _is_param(name: str) -> bool:
    return name.startswith(_PARAM_PREFIXES) or name.endswith(_PARAM_SUFFIXES)


# Methods whose ``self.x = ...`` assignments are construction or sanitisation,
# not integration dynamics. Everything else (step, simulate, reset, helpers) that
# assigns an instance field is treated as evidence the field is state.
_NON_DYNAMICS_METHOD_PREFIXES = (
    "validate",
    "raise",
    "check",
    "finite",
    "ensure",
    "assert",
    "nonneg",
    "non_negative",
    "positive",
    "clamp",
    "sanitize",
    "guard",
)


def _is_non_dynamics_method(name: str) -> bool:
    if name in ("__init__", "__post_init__"):
        return True
    return name.lstrip("_").startswith(_NON_DYNAMICS_METHOD_PREFIXES)


def _dynamic_state_fields(cls: type) -> frozenset[str]:
    """Return instance fields assigned outside construction and validation.

    A field assigned (``self.x = ...`` or ``self.x += ...``, including tuple
    unpacking) inside a dynamics method — step, simulate, reset, or a helper they
    call — is an integration state variable. Assignments confined to
    ``__init__``/``__post_init__`` or a validation/clamp helper are construction or
    sanitisation, not state. This reads the model's actual behaviour rather than
    guessing from the field name, which the name heuristics cannot do reliably for
    internal currents (i1, i2), traces (inh_trace), and adaptation variables.
    """
    try:
        source = textwrap.dedent(inspect.getsource(cls))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError):
        return frozenset()
    classdef = next((node for node in tree.body if isinstance(node, ast.ClassDef)), None)
    if classdef is None:
        return frozenset()
    fields: set[str] = set()
    for node in classdef.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if _is_non_dynamics_method(node.name):
            continue
        for sub in ast.walk(node):
            targets: list[ast.expr] = []
            if isinstance(sub, ast.Assign):
                targets = list(sub.targets)
            elif isinstance(sub, ast.AugAssign):
                targets = [sub.target]
            for target in targets:
                for leaf in ast.walk(target):
                    if (
                        isinstance(leaf, ast.Attribute)
                        and isinstance(leaf.value, ast.Name)
                        and leaf.value.id == "self"
                    ):
                        fields.add(leaf.attr)
    return frozenset(fields)


def _load_v1_schema(module: str) -> dict[str, Any]:
    """Return the curated v1 schema for a module, or an empty mapping."""
    from sc_neurocore.neurons.universal_dsl import load_schema

    try:
        return load_schema(module)
    except FileNotFoundError:
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
    ValueError
        If ``class_name`` is not a public Python identifier.
    KeyError
        If ``class_name`` is not registered.
    """
    if not class_name.isidentifier() or class_name.startswith("_"):
        raise ValueError("model class name must be a public Python identifier")
    if class_name not in _CLASS_TO_MODULE:
        raise KeyError(class_name)
    module = _CLASS_TO_MODULE[class_name]
    cls = _load_class(class_name)
    family, category = model_family(class_name) or ("", "")
    v1 = _load_v1_schema(module)
    v1_meta = v1.get("metadata", {}) if isinstance(v1.get("metadata"), Mapping) else {}
    v1_state = v1.get("state", {}) if isinstance(v1.get("state"), Mapping) else {}
    v1_params = v1.get("parameters", {}) if isinstance(v1.get("parameters"), Mapping) else {}

    specs = _field_specs(cls)
    dyn_state = _dynamic_state_fields(cls)
    state: dict[str, Any] = {}
    parameters: dict[str, Any] = {}
    dt = 0.1
    for name, default in specs:
        if name == "dt":
            dt = default
            continue
        if name in v1_state:
            state[name] = {"init": float(v1_state[name]) if _is_number(v1_state[name]) else default}
        elif name in v1_params:
            # A curated v1 schema is authoritative: it keeps its declared parameters.
            parameters[name] = {"default": default, "unit": "", "meaning": ""}
        elif name in dyn_state:
            # Assigned by the model's dynamics — an integration state variable.
            state[name] = {"init": default}
        elif _is_param(name) or not _is_state_var(name):
            parameters[name] = {"default": default, "unit": "", "meaning": ""}
        else:
            state[name] = {"init": default}
    # No fabricated fallback: a model whose integration state is an internal
    # accumulator (rate, statistical, and generator models) exposes no numeric
    # state field, so its declared state stays empty until curated rather than
    # inventing a membrane potential that does not exist.

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
            "family": family,
            "category": category,
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
    """Return a validated :class:`ModelDescriptor` skeleton for a model.

    Parameters
    ----------
    class_name:
        Registered public model class name to introspect.

    Returns
    -------
    ModelDescriptor
        Parsed descriptor generated from the model code and any curated v1
        schema fields.

    Raises
    ------
    ValueError
        If ``class_name`` is not a public Python identifier.
    KeyError
        If ``class_name`` is not registered.
    """
    return parse_model_descriptor(generate_descriptor_payload(class_name))


_PARAMETER_CURATION_KEYS = ("unit", "range", "biological_range", "meaning")
_STATE_CURATION_KEYS = ("unit", "meaning")
_METADATA_CURATION_KEYS = (
    "display_name",
    "family",
    "category",
    "intended_use",
    "hardware_fit",
    "behavior_tags",
)


def merge_descriptor_payloads(
    curated: Mapping[str, Any],
    regenerated: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge a curated descriptor onto a freshly regenerated one.

    Structural fields (which parameters and state variables exist, their
    defaults and initial values, the timestep) always follow the regenerated
    payload, which is read from the model code — so the corpus can never drift
    from the implementation. Curation fields (parameter units/ranges/meaning,
    state semantics, taxonomy, the backend matrix, reproducibility, notes,
    validation evidence, silicon evidence anchors, and any richer provenance,
    dynamics, or display fields) are preserved from the curated payload. The
    result is the regenerated payload with curation overlaid, ready to be
    re-serialised.

    Parameters
    ----------
    curated:
        The existing on-disk descriptor payload (may carry curation).
    regenerated:
        A freshly generated payload from :func:`generate_descriptor_payload`.

    Returns
    -------
    dict[str, Any]
        The merged descriptor payload.
    """
    merged: dict[str, Any] = {key: _copy(value) for key, value in regenerated.items()}
    cur_meta = _mapping(curated.get("metadata"))
    merged_meta = merged.setdefault("metadata", {})
    for key in _METADATA_CURATION_KEYS:
        value = cur_meta.get(key)
        if value not in (None, "", []):
            merged_meta[key] = _copy(value)
    for key in ("biophysical_detail", "maturity", "summary"):
        value = cur_meta.get(key)
        if isinstance(value, str) and value:
            merged_meta[key] = value

    cur_params = _mapping(curated.get("parameters"))
    for name, spec in merged.get("parameters", {}).items():
        overlay = _mapping(cur_params.get(name))
        for key in _PARAMETER_CURATION_KEYS:
            if key in overlay:
                spec[key] = _copy(overlay[key])

    cur_state = _mapping(curated.get("state"))
    for name, spec in merged.get("state", {}).items():
        overlay = _mapping(cur_state.get(name))
        for key in _STATE_CURATION_KEYS:
            if key in overlay:
                spec[key] = _copy(overlay[key])

    cur_prov = _mapping(curated.get("provenance"))
    if cur_prov:
        merged_prov = {**_mapping(merged.get("provenance")), **cur_prov}
        merged["provenance"] = merged_prov
    cur_dyn = _mapping(curated.get("dynamics"))
    if cur_dyn:
        merged["dynamics"] = {**_mapping(merged.get("dynamics")), **cur_dyn}
    for section in ("backends", "reproducibility", "validation", "silicon"):
        cur_section = _mapping(curated.get(section))
        if cur_section:
            # Evidence facets are curated, never regenerated from constructor
            # inspection. Preserve them wholesale so dual-axis readiness (S4/S5
            # and H0-H5) cannot be wiped by a structural corpus refresh.
            merged[section] = _copy(cur_section)
    cur_doc = _mapping(curated.get("documentation"))
    notes = cur_doc.get("notes")
    if isinstance(notes, str) and notes:
        merged.setdefault("documentation", {})["notes"] = notes
    return merged


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy(item) for item in value]
    return value


__all__ = [
    "generate_descriptor",
    "generate_descriptor_payload",
    "merge_descriptor_payloads",
]
