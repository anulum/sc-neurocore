# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Engine ABI inventory comparison

"""Read and compare engine interface inventories without loading stored pickles."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TypeAlias

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]

IDENTITY_FIELDS = ("__module__", "__qualname__", "__name__", "__text_signature__")
CALLABLE_KINDS = frozenset(
    (
        "type",
        "function",
        "builtin_function_or_method",
        "method",
        "method_descriptor",
        "classmethod_descriptor",
        "wrapper_descriptor",
        "method-wrapper",
        "staticmethod",
    )
)


def validate_json(value: object) -> JsonValue:
    """Validate an inventory's JSON values and reject non-finite numbers.

    Parameters
    ----------
    value : object
        Decoded JSON value.

    Returns
    -------
    JsonValue
        Value with validated string keys and finite scalar numbers.

    Raises
    ------
    ValueError
        A value or dictionary key is outside the inventory format.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, list):
        return [validate_json(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: validate_json(item) for key, item in value.items()}
    raise ValueError("inventory must contain JSON values with finite numbers and string keys")


def _symbol_record(value: JsonValue, *, global_symbol: bool) -> None:
    """Require metadata emitted for every symbol, callable and captured class.

    A null identity or an authored introspection failure remains observable.
    Missing fields cannot stand for such an observation. Class-member records
    contain direct symbol metadata rather than recursively captured classes.
    """
    if not isinstance(value, dict) or not isinstance(value.get("kind"), str):
        raise ValueError("engine ABI symbol has no kind")
    if any(
        field not in value or not isinstance(value[field], (str, type(None)))
        for field in IDENTITY_FIELDS
    ):
        raise ValueError("engine ABI symbol requires complete identity metadata")
    callable_record = (
        value["kind"] in CALLABLE_KINDS or "signature" in value or "signature_error" in value
    )
    captured_class = global_symbol and (value["kind"] == "type" or "members" in value)
    if callable_record or captured_class:
        if (
            "signature" not in value
            or "signature_error" not in value
            or not isinstance(value["signature"], (str, type(None)))
            or not isinstance(value["signature_error"], (str, type(None)))
            or (value["signature"] is None) == (value["signature_error"] is None)
        ):
            raise ValueError("engine ABI callable requires a signature or observed failure")
    if captured_class:
        members = value.get("members")
        if not isinstance(members, dict):
            raise ValueError("engine ABI class requires member metadata")
        for member in members.values():
            _symbol_record(member, global_symbol=False)
    if global_symbol and (value["kind"] in CALLABLE_KINDS or captured_class):
        reference = value.get("global_reference")
        aliases = value.get("aliases")
        if (
            not isinstance(reference, dict)
            or set(reference) != {"same_object", "error"}
            or type(reference["same_object"]) is not bool
            or not isinstance(reference["error"], (str, type(None)))
            or (reference["same_object"] and reference["error"] is not None)
        ):
            raise ValueError("engine ABI global requires reference-resolution metadata")
        if (
            not isinstance(aliases, list)
            or not aliases
            or not all(isinstance(alias, str) and alias for alias in aliases)
            or len(aliases) != len(set(aliases))
        ):
            raise ValueError("engine ABI global requires unique named aliases")


def _unique_json_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    """Retain each recorded field exactly once before validating its value."""
    fields: dict[str, JsonValue] = {}
    for name, value in pairs:
        if name in fields:
            raise ValueError("engine ABI capture contains duplicate object keys")
        fields[name] = value
    return fields


def read_inventory(path: Path) -> dict[str, JsonValue]:
    """Read the versioned interface payload of a captured engine inventory.

    Parameters
    ----------
    path : pathlib.Path
        JSON capture produced by ``engine_abi_inventory.py``.

    Returns
    -------
    dict of str to JsonValue
        Interface payload, excluding measured environment provenance.

    Raises
    ------
    ValueError
        The capture has duplicate object keys, an unsupported schema or lacks
        its interface fields.
    """
    raw = validate_json(
        json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object)
    )
    if not isinstance(raw, dict) or type(raw.get("schema")) is not int or raw.get("schema") != 1:
        raise ValueError("engine ABI capture requires schema 1")
    inventory = raw.get("inventory")
    if not isinstance(inventory, dict):
        raise ValueError("engine ABI capture has no inventory object")
    modules = inventory.get("modules")
    exports = inventory.get("facade_exports")
    if not isinstance(modules, dict) or not isinstance(exports, list):
        raise ValueError("engine ABI inventory requires modules and facade_exports")
    required = {"sc_neurocore_engine", "sc_neurocore_engine.sc_neurocore_engine"}
    if modules.keys() != required or not exports or not all(isinstance(n, str) for n in exports):
        raise ValueError("engine ABI inventory requires both namespaces and nonempty named exports")
    if len(exports) != len(set(exports)):
        raise ValueError("engine ABI facade_exports must contain unique names")
    for namespace, symbols in modules.items():
        if not isinstance(symbols, dict) or "__version__" not in symbols:
            raise ValueError(f"engine ABI namespace {namespace} has no versioned symbol inventory")
        for name, details in symbols.items():
            _symbol_record(details, global_symbol=True)
    facade = modules["sc_neurocore_engine"]
    if not isinstance(facade, dict) or any(name not in facade for name in exports):
        raise ValueError("engine ABI facade_exports names an absent symbol")
    return inventory


def differences(before: JsonValue, after: JsonValue, path: str = "inventory") -> list[str]:
    """Describe every removed, added or changed interface field.

    Parameters
    ----------
    before, after : JsonValue
        Validated interface trees to compare.
    path : str, default="inventory"
        Root used in diagnostic paths.

    Returns
    -------
    list of str
        Sorted field changes; an empty list proves equality of both trees.
    """
    if type(before) is not type(after):
        return [f"{path}: changed {before!r} -> {after!r}"]
    if isinstance(before, dict) and isinstance(after, dict):
        changes = [f"{path}.{key}: removed" for key in sorted(before.keys() - after.keys())]
        changes += [f"{path}.{key}: added" for key in sorted(after.keys() - before.keys())]
        for key in sorted(before.keys() & after.keys()):
            changes.extend(differences(before[key], after[key], f"{path}.{key}"))
        return changes
    if before != after:
        return [f"{path}: changed {before!r} -> {after!r}"]
    return []
