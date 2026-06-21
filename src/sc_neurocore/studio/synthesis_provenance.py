# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio synthesis target provenance

"""Path-free target provenance contracts for Studio synthesis results."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias, cast

from sc_neurocore.studio.evidence_classification import (
    StudioEvidenceClassification,
    StudioEvidenceStatus,
    validate_studio_evidence_classification,
    validate_studio_evidence_status,
)

STUDIO_SYNTHESIS_TARGET_PROVENANCE_SCHEMA_VERSION = "studio.synthesis-target-provenance.v1"
STUDIO_SYNTHESIS_TARGET_PROVENANCE_MATRIX_SCHEMA_VERSION = (
    "studio.synthesis-target-provenance-matrix.v1"
)

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
ToolStatusValue: TypeAlias = bool | str | None
ToolStatus: TypeAlias = Mapping[str, ToolStatusValue]
ToolStatusMap: TypeAlias = Mapping[str, ToolStatus]
TargetConfig: TypeAlias = Mapping[str, str | None]
TargetConfigMap: TypeAlias = Mapping[str, TargetConfig]
CapacityMap: TypeAlias = Mapping[str, Mapping[str, int]]


@dataclass(frozen=True, slots=True)
class StudioSynthesisToolProvenance:
    """Version and availability evidence for one synthesis-related tool.

    Parameters
    ----------
    key:
        Stable API key for the tool status entry.
    executable:
        Command name used by the Studio backend.
    role:
        Tool role in the synthesis workflow.
    available:
        Whether the tool was detected by the backend.
    version:
        First version line returned by the tool, when available.
    """

    key: str
    executable: str
    role: str
    available: bool
    version: str | None

    def to_public_dict(self) -> dict[str, JsonValue]:
        """Return a path-free public tool provenance payload."""

        return {
            "available": self.available,
            "executable": self.executable,
            "key": self.key,
            "role": self.role,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class StudioSynthesisTargetProvenance:
    """Path-free provenance for one Studio synthesis target.

    Parameters
    ----------
    target:
        Studio target identifier.
    capacity:
        Static capacity metadata used for resource utilisation.
    synthesis_command:
        Yosys synthesis command selected for the target.
    pnr_tool:
        Place-and-route command for the target, when one is configured.
    device:
        Device selector passed to the PnR tool, when configured.
    tools:
        Required tool provenance records for this target.
    evidence_classification:
        Stable evidence lane label for synthesis target provenance.
    status:
        Terminal status for this synthesis target provenance object.
    """

    target: str
    capacity: dict[str, int]
    synthesis_command: str
    pnr_tool: str | None
    device: str | None
    tools: tuple[StudioSynthesisToolProvenance, ...]
    evidence_classification: StudioEvidenceClassification = "synthesis"
    status: StudioEvidenceStatus = "completed"

    def to_public_dict(self) -> dict[str, JsonValue]:
        """Return the public, path-free target provenance payload."""

        tools = [tool.to_public_dict() for tool in self.tools]
        return {
            "capacity": cast(dict[str, JsonValue], self.capacity),
            "device": self.device,
            "evidence_classification": validate_studio_evidence_classification(
                self.evidence_classification
            ),
            "pnr_ready": all(
                tool.available for tool in self.tools if tool.role == "place_and_route"
            ),
            "pnr_tool": self.pnr_tool,
            "schema_version": STUDIO_SYNTHESIS_TARGET_PROVENANCE_SCHEMA_VERSION,
            "status": validate_studio_evidence_status(self.status),
            "synthesis_command": self.synthesis_command,
            "synthesis_ready": all(
                tool.available for tool in self.tools if tool.role == "synthesis"
            ),
            "target": self.target,
            "tools": cast(list[JsonValue], tools),
        }


def build_synthesis_target_provenance(
    target: str,
    *,
    target_config: TargetConfig,
    capacity: Mapping[str, int],
    tool_status: ToolStatusMap,
) -> StudioSynthesisTargetProvenance:
    """Build provenance metadata for one Studio synthesis target.

    Parameters
    ----------
    target:
        Studio target identifier.
    target_config:
        Target configuration containing synthesis, PnR, and device selectors.
    capacity:
        Static target capacity metadata.
    tool_status:
        Path-free tool detection payload from ``check_tools``.

    Returns
    -------
    StudioSynthesisTargetProvenance
        Path-free target provenance record.

    Raises
    ------
    ValueError
        If the target configuration lacks a synthesis command.
    """

    synthesis_command = target_config.get("synth_cmd")
    if synthesis_command is None:
        raise ValueError(f"Synthesis target '{target}' has no synthesis command.")
    pnr_tool = target_config.get("pnr")
    tools = [
        _tool_provenance(
            key="yosys",
            executable="yosys",
            role="synthesis",
            tool_status=tool_status,
        )
    ]
    if pnr_tool is not None:
        tools.append(
            _tool_provenance(
                key=_pnr_tool_status_key(pnr_tool),
                executable=pnr_tool,
                role="place_and_route",
                tool_status=tool_status,
            )
        )
    return StudioSynthesisTargetProvenance(
        target=target,
        capacity=dict(capacity),
        synthesis_command=synthesis_command,
        pnr_tool=pnr_tool,
        device=target_config.get("device"),
        tools=tuple(tools),
    )


def build_synthesis_target_provenance_matrix(
    *,
    targets: TargetConfigMap,
    capacities: CapacityMap,
    tool_status: ToolStatusMap,
) -> dict[str, JsonValue]:
    """Build a path-free provenance matrix for all Studio synthesis targets.

    Parameters
    ----------
    targets:
        Mapping of target identifiers to backend target configuration.
    capacities:
        Mapping of target identifiers to static capacity metadata.
    tool_status:
        Path-free tool detection payload from ``check_tools``.

    Returns
    -------
    dict[str, JsonValue]
        Matrix payload keyed by target with a stable SHA-256 digest.
    """

    target_payloads = {
        target: build_synthesis_target_provenance(
            target,
            target_config=config,
            capacity=capacities.get(target, {}),
            tool_status=tool_status,
        ).to_public_dict()
        for target, config in targets.items()
    }
    matrix_contract: dict[str, JsonValue] = {
        "schema_version": STUDIO_SYNTHESIS_TARGET_PROVENANCE_MATRIX_SCHEMA_VERSION,
        "targets": cast(dict[str, JsonValue], target_payloads),
    }
    matrix_contract["matrix_sha256"] = _sha256_json(matrix_contract)
    return matrix_contract


def _tool_provenance(
    *,
    key: str,
    executable: str,
    role: str,
    tool_status: ToolStatusMap,
) -> StudioSynthesisToolProvenance:
    """Build one tool provenance record from path-free tool status."""

    status = tool_status.get(key, {})
    return StudioSynthesisToolProvenance(
        key=key,
        executable=executable,
        role=role,
        available=status.get("available") is True,
        version=_optional_str(status.get("version")),
    )


def _pnr_tool_status_key(executable: str) -> str:
    """Return the Studio tool-status key for a PnR executable."""

    return executable.replace("-", "_")


def _optional_str(value: object) -> str | None:
    """Return a string value or ``None`` for public JSON output."""

    return value if isinstance(value, str) else None


def _sha256_json(payload: Mapping[str, JsonValue]) -> str:
    """Return a stable SHA-256 digest over canonical JSON."""

    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
