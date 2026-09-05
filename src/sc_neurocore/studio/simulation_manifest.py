# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio simulation run manifests

"""Path-free reproducibility manifests for Studio simulation runs.

Version 2 of the manifest separates the digest of the complete raw result from
the digest of the returned payload, and records the observation clock, the
state-layout source and whether the recorded state is the complete declared
state. Version 1 manifests remain readable by evidence consumers as history.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from sc_neurocore.studio.evidence_classification import (
    StudioEvidenceClassification,
    StudioEvidenceStatus,
    validate_studio_evidence_classification,
    validate_studio_evidence_status,
)

STUDIO_SIMULATION_RUN_SCHEMA_VERSION = "studio.simulation-run.v2"
STUDIO_SIMULATION_RUN_SCHEMA_V1 = "studio.simulation-run.v1"
SUPPORTED_SIMULATION_RUN_SCHEMA_VERSIONS: frozenset[str] = frozenset(
    {STUDIO_SIMULATION_RUN_SCHEMA_V1, STUDIO_SIMULATION_RUN_SCHEMA_VERSION}
)

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
SimulationSource: TypeAlias = Literal["ode", "model"]


@dataclass(frozen=True, slots=True)
class StudioSimulationRunManifest:
    """Path-free metadata for one Studio simulation result.

    Parameters
    ----------
    source:
        Simulation surface that produced the result.
    input_sha256:
        SHA-256 digest of the canonical request payload.
    result_sha256:
        SHA-256 digest of the canonical result payload without this manifest.
    raw_sha256:
        SHA-256 digest of the canonical ``raw`` block (full-resolution traces,
        spikes and clock rules), empty when the result carries no raw block.
    dt:
        Effective simulation time step in milliseconds.
    n_steps:
        Number of executed simulation steps.
    sample_count:
        Number of display samples returned in ``time``.
    spike_count:
        Number of spikes detected during the simulation.
    state_variables:
        Sorted names of the state variables recorded at every step.
    raw_included:
        Whether the raw block carries the full-resolution traces.
    layout_source:
        Where the state layout came from (``descriptor``, ``equations`` or
        ``undeclared``), empty for a result without a layout.
    state_custody_complete:
        Whether every declared variable was recorded and nothing undeclared
        changed during the run.
    observation_clock:
        Clock rule of the recorded samples (``post-step``), empty when the
        result carries none.
    evidence_classification:
        Stable evidence lane label for simulation runs.
    status:
        Terminal status for this simulation evidence object.
    """

    source: SimulationSource
    input_sha256: str
    result_sha256: str
    raw_sha256: str
    dt: float
    n_steps: int
    sample_count: int
    spike_count: int
    state_variables: tuple[str, ...]
    raw_included: bool
    layout_source: str
    state_custody_complete: bool
    observation_clock: str
    evidence_classification: StudioEvidenceClassification = "simulation"
    status: StudioEvidenceStatus = "completed"

    def to_public_dict(self) -> dict[str, JsonValue]:
        """Return the public, path-free simulation run manifest."""
        return {
            "dt": self.dt,
            "evidence_classification": validate_studio_evidence_classification(
                self.evidence_classification
            ),
            "input_sha256": self.input_sha256,
            "layout_source": self.layout_source,
            "n_steps": self.n_steps,
            "observation_clock": self.observation_clock,
            "raw_included": self.raw_included,
            "raw_sha256": self.raw_sha256,
            "result_sha256": self.result_sha256,
            "sample_count": self.sample_count,
            "schema_version": STUDIO_SIMULATION_RUN_SCHEMA_VERSION,
            "source": self.source,
            "spike_count": self.spike_count,
            "state_custody_complete": self.state_custody_complete,
            "status": validate_studio_evidence_status(self.status),
            "state_variables": list(self.state_variables),
        }


def build_simulation_run_manifest(
    *,
    source: SimulationSource,
    request_payload: Mapping[str, Any],
    result_payload: Mapping[str, Any],
) -> StudioSimulationRunManifest:
    """Build digest-backed reproducibility metadata for a simulation result.

    Parameters
    ----------
    source:
        Simulation surface that produced the result.
    request_payload:
        Request body used to run the simulation.
    result_payload:
        Result payload after classification, without relying on local paths.

    Returns
    -------
    StudioSimulationRunManifest
        Path-free metadata suitable for UI display, exported JSON, and evidence
        bundles.

    Raises
    ------
    ValueError
        If request or result payloads cannot be encoded as portable JSON.
    """
    states = result_payload.get("states")
    time = result_payload.get("time")
    raw = result_payload.get("raw")
    layout = result_payload.get("state_layout")
    observation = result_payload.get("observation")
    result_without_manifest = {
        key: value for key, value in result_payload.items() if key != "run_metadata"
    }
    recorded: list[str]
    if isinstance(layout, Mapping) and isinstance(layout.get("recorded"), list):
        recorded = sorted(str(name) for name in layout["recorded"])
    elif isinstance(states, dict):
        recorded = sorted(states)
    else:
        recorded = []
    return StudioSimulationRunManifest(
        source=source,
        input_sha256=_sha256_json(request_payload),
        result_sha256=_sha256_json(result_without_manifest),
        raw_sha256=_sha256_json(raw) if isinstance(raw, Mapping) else "",
        dt=_float_field(result_payload, "dt"),
        n_steps=_int_field(result_payload, "n_steps"),
        sample_count=len(time) if isinstance(time, list) else 0,
        spike_count=_int_field(result_payload, "spike_count"),
        state_variables=tuple(recorded),
        raw_included=bool(raw.get("included")) if isinstance(raw, Mapping) else False,
        layout_source=str(layout.get("source", "")) if isinstance(layout, Mapping) else "",
        state_custody_complete=(
            bool(layout.get("complete")) if isinstance(layout, Mapping) else False
        ),
        observation_clock=(
            str(observation.get("clock", "")) if isinstance(observation, Mapping) else ""
        ),
    )


def _sha256_json(payload: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 digest over portable canonical JSON."""
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            default=str,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Simulation manifest payload must be portable JSON.") from exc
    return hashlib.sha256(encoded).hexdigest()


def _float_field(payload: Mapping[str, Any], key: str) -> float:
    """Read a numeric manifest field from a simulation result payload."""
    value = payload.get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _int_field(payload: Mapping[str, Any], key: str) -> int:
    """Read an integer manifest field from a simulation result payload."""
    value = payload.get(key)
    return int(value) if isinstance(value, int) else 0
