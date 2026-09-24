# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — What each silicon operation can do for one catalogue model, and why not

"""State, for one catalogue model, which silicon operations this installation can run.

Each operation is enabled only when its whole chain can execute here, and a
disabled one carries the reason:

* **compile** needs the model's canonical schema with an executable profile
  and a Studio fixed-point format that holds it; the integrators offered are
  the profile's own family.
* **cosimulate** compares the RTL with a generated bit-true C kernel, so it is
  offered per integrator and format only where that kernel mirrors the RTL,
  and only when Icarus Verilog and a C compiler are installed.
* **synthesise** runs on RTL whose co-simulation was bit-exact (the terminal
  refuses any other), so it needs co-simulation and Yosys.
* **place and route** reports timing; it needs synthesis and the target's
  place-and-route tool. A target with none is disabled by name.
* **formal**: the catalogue's bounded formal jobs run in a checkout; the Studio
  has no route that runs them. The job a model has is named so a reader can
  see what it asserts and what it does not establish.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sc_neurocore.neurons.model_catalogue import load_descriptor
from sc_neurocore.neurons.readiness import REPO_ROOT
from sc_neurocore.studio.model_catalogue import _compile_configuration
from sc_neurocore.studio.model_numeric_contracts import bit_true_mirrored
from sc_neurocore.studio.synthesis import _TARGETS, check_tools

MODEL_CAPABILITIES_SCHEMA_VERSION = "sc-neurocore.studio.model-capabilities.v1"
FORMAL_INVENTORY = REPO_ROOT / "hdl" / "formal" / "catalogue" / "inventory.json"
_COSIM_TOOLS = ("iverilog", "vvp", "gcc")


def _operation(enabled: bool, reason: str | None, **detail: Any) -> dict[str, Any]:
    return {"enabled": enabled, "reason": None if enabled else reason, **detail}


def model_capabilities(
    name: str,
    *,
    tool_status: Mapping[str, Mapping[str, object]] | None = None,
    formal_inventory: Path = FORMAL_INVENTORY,
) -> dict[str, Any] | None:
    """Return what each silicon operation can do for ``name`` here, or ``None``.

    Parameters
    ----------
    name:
        A registered catalogue model.
    tool_status:
        An EDA tool snapshot as :func:`~sc_neurocore.studio.synthesis.check_tools`
        returns; taken now when omitted.
    formal_inventory:
        The catalogue's formal-job inventory.

    Returns
    -------
    dict or None
        ``None`` for a name the catalogue does not hold.
    """
    descriptor = load_descriptor(name)
    if descriptor is None:
        return None
    tools = check_tools() if tool_status is None else tool_status
    configuration = _compile_configuration(descriptor)
    compile_op = _compile_operation(configuration)
    cosim = _cosim_operation(configuration, compile_op["enabled"])
    synthesis = _synthesis_operation(cosim["enabled"], tools)
    return {
        "schema_version": MODEL_CAPABILITIES_SCHEMA_VERSION,
        "model": name,
        "operations": {
            "compile": compile_op,
            "cosimulate": cosim,
            "synthesise": synthesis,
            "place_and_route": _pnr_operation(synthesis["enabled"], tools),
            "formal": _formal_operation(name, formal_inventory),
        },
    }


def _compile_operation(configuration: Mapping[str, Any] | None) -> dict[str, Any]:
    if configuration is None:
        return _operation(
            False,
            "the model has no canonical schema with an executable profile for the compiler",
        )
    if not configuration["q_formats"]:
        return _operation(
            False,
            "no Studio fixed-point format holds the model; numeric_contracts says why",
            integrators=list(configuration["integrators"]),
        )
    return _operation(
        True,
        None,
        integrators=list(configuration["integrators"]),
        q_formats=list(configuration["q_formats"]),
    )


def _cosim_operation(
    configuration: Mapping[str, Any] | None,
    compiles: bool,
) -> dict[str, Any]:
    if configuration is None or not compiles:
        return _operation(False, "the model does not compile")
    combinations = [
        {
            "integrator": integrator,
            "q_format": q_format,
            "mirrored": bit_true_mirrored(configuration["schema_name"], integrator, q_format),
        }
        for integrator in configuration["cosim_integrators"]
        for q_format in configuration["q_formats"]
    ]
    unmirrored = [
        integrator
        for integrator in configuration["integrators"]
        if integrator not in configuration["cosim_integrators"]
    ]
    if not any(combination["mirrored"] for combination in combinations):
        return _operation(
            False,
            "no bit-true C kernel mirrors this model's RTL for any offered integrator",
            unmirrored_integrators=unmirrored,
        )
    missing = [tool for tool in _COSIM_TOOLS if shutil.which(tool) is None]
    if missing:
        return _operation(
            False,
            f"co-simulation needs {', '.join(missing)}, which is not installed",
            combinations=combinations,
        )
    return _operation(True, None, combinations=combinations, unmirrored_integrators=unmirrored)


def _synthesis_operation(
    cosimulates: bool, tools: Mapping[str, Mapping[str, object]]
) -> dict[str, Any]:
    if not cosimulates:
        return _operation(False, "synthesis runs only on RTL whose co-simulation was bit-exact")
    if not tools.get("yosys", {}).get("available"):
        return _operation(False, "Yosys is not installed")
    return _operation(True, None, targets=list(_TARGETS))


def _pnr_operation(synthesises: bool, tools: Mapping[str, Mapping[str, object]]) -> dict[str, Any]:
    if not synthesises:
        return _operation(False, "place and route follows synthesis")
    enabled: list[str] = []
    disabled: dict[str, str] = {}
    for target, configuration in _TARGETS.items():
        tool = configuration.get("pnr")
        if tool is None:
            disabled[target] = "this target has no place-and-route tool in the Studio flow"
        elif not tools.get(str(tool).replace("-", "_"), {}).get("available"):
            disabled[target] = f"{tool} is not installed"
        else:
            enabled.append(target)
    return _operation(
        bool(enabled),
        "no target's place-and-route tool is installed",
        targets=enabled,
        disabled_targets=disabled,
    )


def _formal_operation(name: str, inventory: Path) -> dict[str, Any]:
    reason = (
        "the Studio has no route that runs a formal job; the catalogue's jobs run in a "
        "checkout with SymbiYosys"
    )
    if not inventory.is_file():
        return _operation(False, reason, catalogue_job=None)
    jobs = json.loads(inventory.read_text(encoding="utf-8"))["jobs"]
    job = next((entry for entry in jobs if entry.get("class") == name), None)
    if job is None:
        return _operation(False, reason, catalogue_job=None)
    return _operation(
        False,
        reason,
        catalogue_job={
            "module": job["module"],
            "claim": job["claim"],
            "q_format": job["q_format"],
            "depth": job["depth"],
            "properties": list(job["properties"]),
            "not_established": list(job["not_established"]),
        },
    )


__all__ = ["MODEL_CAPABILITIES_SCHEMA_VERSION", "model_capabilities"]
