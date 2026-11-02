# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Assertion-enable cover model

"""Derive covers of assertion enables from an already prepared Yosys model."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass

from tools.formal_evidence_validation import object_fields, require


@dataclass(frozen=True)
class EnableModel:
    """Retain the auxiliary graph and its assertion-to-cover identities.

    Parameters
    ----------
    serialized : str
        Deterministically serialized auxiliary Yosys JSON.
    top : str
        Original top module.
    goals : dict of str to str
        Auxiliary cover names and their original assertion identities.
    """

    serialized: str
    top: str
    goals: dict[str, str]


def prepared_top(design: dict[str, object]) -> tuple[str, dict[str, object]]:
    """Select the native flattened top and refuse unhandled formal cells.

    Parameters
    ----------
    design : dict of str to object
        Native Yosys JSON after flattening the primary prepared model.

    Returns
    -------
    tuple of str and dict
        Unique top name and its cells; unused modules never supply goals.

    Raises
    ------
    ValueError
        If the model lacks a unique flattened top or has unhandled checks.
    """
    modules = object_fields(design.get("modules"))
    tops = [
        name
        for name, value in modules.items()
        if object_fields(object_fields(value).get("attributes")).get("top")
        == "00000000000000000000000000000001"
    ]
    require(len(tops) == 1, "Prepared model must have exactly one top.")
    top = tops[0]
    module = object_fields(modules[top])
    cells = object_fields(module.get("cells"))
    require(
        all(str(object_fields(cell).get("type")).startswith("$") for cell in cells.values()),
        "Prepared top must be flattened by native Yosys.",
    )
    require(
        all(
            object_fields(cell).get("type") not in ("$check", "$live", "$fair", "$print")
            for cell in cells.values()
        ),
        "Prepared model contains an unsupported formal or print cell.",
    )
    return top, cells


def enable_model(serialized: str) -> EnableModel:
    """Replace each assertion by a cover of its unchanged enable connection.

    Parameters
    ----------
    serialized : str
        Native Yosys JSON of the same primary ``design_prep.il``, flattened
        by Yosys. All original assumptions, state, initialization and logic
        remain unchanged. Existing covers are removed from the auxiliary.

    Returns
    -------
    EnableModel
        One auxiliary cover per original assertion, with constant-true A
        and the assertion's exact EN. This checks reachability, not strength.

    Raises
    ------
    ValueError
        If the prepared model has no unique top, no assertions, unsupported
        hierarchy or malformed formal cells.
    """
    design = deepcopy(object_fields(json.loads(serialized)))
    top, cells = prepared_top(design)
    goals: dict[str, str] = {}
    for name, value in sorted(list(cells.items())):
        cell = object_fields(value)
        kind = cell.get("type")
        if kind == "$cover":
            del cells[name]
        elif kind == "$assert":
            connections = object_fields(cell.get("connections"))
            require(
                set(connections) == {"A", "EN"}
                and isinstance(connections["A"], list)
                and len(connections["A"]) == 1
                and isinstance(connections["EN"], list)
                and len(connections["EN"]) == 1,
                "Assertion must have scalar A and EN.",
            )
            goal = f"scn_assertion_enable_{len(goals):06d}"
            require(goal not in cells, "Auxiliary cover name collides with a native cell.")
            del cells[name]
            connections["A"] = ["1"]
            cell["connections"] = connections
            cell["type"] = "$cover"
            attributes = object_fields(cell.get("attributes"))
            attributes["hdlname"] = goal
            cell["attributes"] = attributes
            cells[goal] = cell
            goals[goal] = f"{top}/{name}"
    require(bool(goals), "Prepared proof has zero assertions; non-vacuity is unqualified.")
    modules = object_fields(design["modules"])
    module = object_fields(modules[top])
    module["cells"] = cells
    modules[top] = module
    design["modules"] = modules
    return EnableModel(json.dumps(design, sort_keys=True, indent=2) + "\n", top, goals)
