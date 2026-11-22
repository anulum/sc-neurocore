# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — What the emitted RTL of one catalogue model is guaranteed by

"""Classify what the emitted RTL of one catalogue model is guaranteed by.

:func:`model_guarantee_summary` uses the classes of
:data:`~sc_neurocore.compiler.guarantee_summary.GUARANTEE_CLASSES`: the
co-simulation is ``CO-SIMULATED ONLY``; the model's catalogue formal job
(:func:`formal_guarantee`) is ``FORMAL`` only when its RTL file is
byte-identical to the emitted RTL, ``PROVEN AT OTHER PARAMETERS`` otherwise,
and ``NONE`` when the model has no job. No formal job is run here. The
readiness labels (:func:`readiness_labels`) come from the sealed
``verified_readiness.json``. The hardware path's guarantee stage consumes this
classification; it does not own it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sc_neurocore.compiler.guarantee_summary import GUARANTEE_CLASSES
from sc_neurocore.studio.model_capabilities import FORMAL_INVENTORY
from sc_neurocore.studio.model_compile_configuration import ResolvedModelCompileConfiguration
from sc_neurocore.studio.readiness_seal import SEAL_PATH, sealed_detail

MODEL_GUARANTEE_SCHEMA_VERSION = "sc-neurocore.studio.model-guarantee-summary.v1"

_FORMAL_COMPARED = (("module", "module"), ("profile", "schema"), ("q_format", "q_format"))


def model_guarantee_summary(
    configuration: ResolvedModelCompileConfiguration,
    verilog: str,
    cosim_report: Mapping[str, Any],
    *,
    formal_inventory: Path = FORMAL_INVENTORY,
    readiness_seal: Path = SEAL_PATH,
) -> dict[str, Any]:
    """Classify what the emitted RTL of one model is guaranteed by.

    Parameters
    ----------
    configuration : ResolvedModelCompileConfiguration
        The resolved model configuration.
    verilog : str
        The RTL emitted for it.
    cosim_report : Mapping
        Its bit-exact ``studio.cosim-parity.v1`` report.
    formal_inventory : Path, optional
        The catalogue formal inventory.
    readiness_seal : Path, optional
        The sealed verified-readiness record.

    Returns
    -------
    dict
        Schema, model, RTL digest, the co-simulation and formal guarantee
        entries, the readiness labels and the number of entries per class of
        :data:`~sc_neurocore.compiler.guarantee_summary.GUARANTEE_CLASSES`.
    """
    boundary = cosim_report["boundary"]
    cosim_entry: dict[str, Any] = {
        "aspect": "co-simulation",
        "class": "CO-SIMULATED ONLY",
        "compared": boundary["compared"],
        "statement": boundary["statement"],
        "not_covered": list(boundary["not_covered"]),
        "q_format": configuration.q_format.q_label,
        "stimulus": cosim_report["stimulus"],
        "stress_sample_count": cosim_report["stress"]["sample_count"],
        "report": "cosim/report.json",
    }
    run_parameters: dict[str, object] = {
        "module": configuration.module_name,
        "schema": configuration.schema_name,
        "q_format": configuration.q_format.q_label,
        "integrator": configuration.integrator,
        "dt": configuration.dt,
        "params": dict(configuration.params),
    }
    formal_entry = formal_guarantee(
        configuration.model_name, verilog, run_parameters, inventory=formal_inventory
    )
    counts = {name: 0 for name in GUARANTEE_CLASSES}
    for entry in (cosim_entry, formal_entry):
        counts[entry["class"]] += 1
    return {
        "schema": MODEL_GUARANTEE_SCHEMA_VERSION,
        "model": configuration.model_name,
        "module_name": configuration.module_name,
        "rtl_sha256": _sha256(verilog.encode("utf-8")),
        "guarantees": [cosim_entry, formal_entry],
        "readiness": readiness_labels(configuration.model_name, readiness_seal),
        "classes": counts,
    }


def formal_guarantee(
    model_name: str,
    verilog: str,
    run_parameters: Mapping[str, object],
    *,
    inventory: Path = FORMAL_INVENTORY,
) -> dict[str, Any]:
    """Classify the catalogue formal job of a model against the emitted RTL.

    The job is never run here: the inventory states what it checks. A job
    covers this RTL only when its RTL file holds exactly the bytes the catalogue
    emitter (``tools/emit_catalogue_formal.py``) writes for this RTL: the
    compiler's text, terminated by one newline when it does not end with one.
    Both digests are recorded.

    Parameters
    ----------
    model_name : str
        The catalogue model.
    verilog : str
        The emitted RTL.
    run_parameters : Mapping
        This run's ``module``, ``schema`` and ``q_format`` (compared with the
        job's ``module``, ``profile`` and ``q_format``) and its other settings.
    inventory : Path, optional
        ``hdl/formal/catalogue/inventory.json``; job RTL files are read beside it.

    Returns
    -------
    dict
        A ``FORMAL``, ``PROVEN AT OTHER PARAMETERS`` or ``NONE`` entry with the
        reason; the first two carry the job and both parameter sets.
    """
    entry: dict[str, Any] = {"aspect": "formal", "ran_here": False}
    if not inventory.is_file():
        return {
            **entry,
            "class": "NONE",
            "inventory_available": False,
            "reason": (
                f"the formal inventory ({inventory.name}) is not available in this "
                "installation, so no catalogue job can be named"
            ),
        }
    document = json.loads(inventory.read_text(encoding="utf-8"))
    jobs = [*document.get("jobs", []), *document.get("retained_jobs", [])]
    job = next((item for item in jobs if item.get("class") == model_name), None)
    if job is None:
        return {
            **entry,
            "class": "NONE",
            "inventory_available": True,
            "reason": f"the catalogue has no formal job for {model_name}",
        }
    job_rtl = inventory.parent / f"{job['module']}.v"
    job_bytes = job_rtl.read_bytes() if job_rtl.is_file() else None
    rtl_bytes = verilog.encode("utf-8")
    as_written = (verilog if verilog.endswith("\n") else verilog + "\n").encode("utf-8")
    identical = job_bytes == as_written
    proved = {"module": job["module"], "profile": job["profile"], "q_format": job["q_format"]}
    differs = [
        job_key
        for job_key, run_key in _FORMAL_COMPARED
        if proved[job_key] != run_parameters.get(run_key)
    ]
    if job_bytes is None:
        return {
            **entry,
            "class": "NONE",
            "inventory_available": True,
            "reason": (
                f"the catalogue names job {job['module']} for {model_name}, but its RTL file "
                f"{job_rtl.name} is not available, so nothing it proves can be attached to "
                "this RTL"
            ),
            "job_rtl": job_rtl.name,
        }
    if identical:
        reason = "the job's RTL file is byte-identical to the emitted RTL as the emitter writes it"
    else:
        reason = "the job's RTL file differs from the emitted RTL" + (
            f" (differs: {', '.join(differs)})" if differs else ""
        )
    result: dict[str, Any] = {
        **entry,
        "class": "FORMAL" if identical else "PROVEN AT OTHER PARAMETERS",
        "inventory_available": True,
        "reason": reason,
        "job": {
            key: job[key]
            for key in (
                "module",
                "origin",
                "profile",
                "q_format",
                "mode",
                "depth",
                "solver",
                "claim",
                "properties",
                "assumptions",
                "not_established",
            )
        },
        "job_rtl": job_rtl.name,
        "job_rtl_sha256": _sha256(job_bytes),
        "rtl_sha256": _sha256(rtl_bytes),
        "compared_rtl_sha256": _sha256(as_written),
        "proved_parameters": proved,
        "run_parameters": dict(run_parameters),
    }
    if not identical:
        result["differs"] = differs
    return result


def readiness_labels(model_name: str, seal: Path = SEAL_PATH) -> dict[str, Any]:
    """Return a model's verified readiness labels from the sealed record.

    Parameters
    ----------
    model_name : str
        The catalogue model.
    seal : Path, optional
        The sealed ``verified_readiness.json``.

    Returns
    -------
    dict
        ``source`` (``sealed`` or ``unsealed``), the verified ``profile``,
        ``science_label`` and ``silicon_label``, every facet's status and, for
        an unsealed record, the reason.
    """
    detail = sealed_detail(model_name, seal)
    labels: dict[str, Any] = {
        "source": detail["source"],
        "seal": seal.name,
        "profile": detail["profile"],
        "science_label": detail["science_label"],
        "silicon_label": detail["silicon_label"],
        "facets": {str(facet["facet"]): str(facet["status"]) for facet in detail["facets"]},
    }
    if "unsealed_reason" in detail:
        labels["unsealed_reason"] = detail["unsealed_reason"]
    return labels


def render_model_guarantee_text(summary: Mapping[str, Any]) -> str:
    """Render a model guarantee summary as short text.

    Parameters
    ----------
    summary : Mapping
        A summary returned by :func:`model_guarantee_summary`.

    Returns
    -------
    str
        One line per guarantee, the readiness labels and the class counts.
    """
    cosim, formal = summary["guarantees"]
    lines = [
        f"Guarantee summary of {summary['model']} ({summary['schema']})",
        f"RTL {summary['module_name']} SHA-256 {summary['rtl_sha256']}",
        f"co-simulation: CO-SIMULATED ONLY ({cosim['compared']} at {cosim['q_format']}: "
        f"{cosim['statement']}; not covered: {'; '.join(cosim['not_covered'])})",
    ]
    if "job" in formal:
        job = formal["job"]
        lines.append(
            f"formal: {formal['class']} {job['module']} ({job.get('claim')}, "
            f"{job.get('mode')} depth {job.get('depth')}, proved at "
            f"{_pairs(formal['proved_parameters'])}; this run "
            f"{_pairs(formal['run_parameters'])}; {formal['reason']}; not run here)"
        )
    else:
        lines.append(f"formal: NONE ({formal['reason']})")
    readiness = summary["readiness"]
    lines.append(
        f"readiness ({readiness['source']}): {readiness['science_label']} / "
        f"{readiness['silicon_label']}"
        + (f" ({readiness['unsealed_reason']})" if "unsealed_reason" in readiness else "")
    )
    lines.append("Classes: " + ", ".join(f"{k} {v}" for k, v in summary["classes"].items()))
    return "\n".join(lines) + "\n"


def _pairs(values: Mapping[str, object]) -> str:
    return ", ".join(f"{key}={value}" for key, value in values.items())


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "MODEL_GUARANTEE_SCHEMA_VERSION",
    "formal_guarantee",
    "model_guarantee_summary",
    "readiness_labels",
    "render_model_guarantee_text",
]
