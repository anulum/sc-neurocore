# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evidence bundle process ingress

"""Transfer bounded evidence snapshots without giving compute a ledger reader."""

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, JsonValue

from sc_neurocore.studio.platform.evidence_bundle import (
    StudioArtifactReader,
    write_studio_evidence_bundle,
)
from sc_neurocore.studio.platform.evidence_limits import validate_evidence_input_limit
from sc_neurocore.studio.platform.jobs import StudioJobContext
from sc_neurocore.studio.platform.jobs_models import (
    StudioJobArtifact,
    StudioJobArtifactPayload,
    StudioJobRecord,
)
from sc_neurocore.studio.platform.jobs_snapshot import decode_job_snapshot


class EvidenceInputLimitExceeded(ValueError):
    """Aggregate snapshot metadata and declared source bytes exceed policy."""


class _BundleInputs(BaseModel):
    """Exact non-executable writer inputs; clocks/readers are never delegated."""

    model_config = ConfigDict(extra="forbid", strict=True)
    project_payload: dict[str, JsonValue] | None
    simulation_payloads: list[dict[str, JsonValue]]
    analysis_payloads: list[dict[str, JsonValue]]
    model_scan_payloads: list[dict[str, JsonValue]]
    weight_restore_payloads: list[dict[str, JsonValue]]
    weight_restore_attach_payloads: list[dict[str, JsonValue]]
    default_flow_runs: list[dict[str, JsonValue]]
    default_flow_attestations: list[dict[str, JsonValue]]
    audit_export: dict[str, JsonValue] | None
    command_replay: dict[str, JsonValue] | None


class _Envelope(BaseModel):
    """Complete source records and immutable API-selected input ceiling."""

    model_config = ConfigDict(extra="forbid", strict=True)
    inputs: _BundleInputs
    records: list[dict[str, JsonValue]]
    max_input_bytes: int


def _seed_entries(
    records: Sequence[StudioJobRecord],
) -> Iterator[tuple[str, str, StudioJobArtifact]]:
    for record_index, record in enumerate(records):
        for artifact_index, artifact in enumerate(record.artifacts):
            yield f"bundle-{record_index}-{artifact_index}.bin", record.job_id, artifact


def _check_budget(
    payload: Mapping[str, object], records: Sequence[StudioJobRecord], limit: int
) -> None:
    total = 0
    for _, _, artifact in _seed_entries(records):
        if artifact.size_bytes < 0:
            raise ValueError("Studio evidence artifact declaration has negative size.")
        total += artifact.size_bytes
        if total > limit:
            raise EvidenceInputLimitExceeded("Studio evidence input byte limit exceeded.")
    for chunk in json.JSONEncoder(sort_keys=True, allow_nan=False).iterencode(dict(payload)):
        total += len(chunk.encode("utf-8"))
        if total > limit:
            raise EvidenceInputLimitExceeded("Studio evidence input byte limit exceeded.")


class EvidenceSeedInputs(Mapping[str, bytes]):
    """Load one verified source artifact at a time while the manager writes seeds.

    This mapping retains only declarations, never a cache of all binary payloads.
    The manager's existing per-seed limit remains independent of aggregate policy.

    Parameters
    ----------
    records : sequence of StudioJobRecord
        Captured records whose ordered artifacts define deterministic seed names.
    reader : callable
        Trusted API-side artifact reader. Each lookup verifies returned metadata,
        size and digest; construction does not read payloads or write files.
    """

    def __init__(self, records: Sequence[StudioJobRecord], reader: StudioArtifactReader) -> None:
        self._entries = {
            name: (job_id, artifact) for name, job_id, artifact in _seed_entries(records)
        }
        self._reader = reader

    def __len__(self) -> int:
        """Return the declared seed count without reading source files."""
        return len(self._entries)

    def __iter__(self) -> Iterator[str]:
        """Yield deterministic seed names in source-record and artifact order."""
        return iter(self._entries)

    def __getitem__(self, name: str) -> bytes:
        """Read and verify one seed; reject unknown names or changed source bytes.

        Parameters
        ----------
        name : str
            Deterministic seed key from this mapping's iterator.

        Returns
        -------
        bytes
            Verified source payload, not retained in an internal cache.

        Raises
        ------
        KeyError
            The seed name is unknown.
        ValueError
            Reader metadata or bytes disagree with the captured declaration.
        StudioJobArtifactUnavailable
            The source reader cannot supply an intact declared artifact.
        OSError
            Reading the source file fails.
        """
        job_id, artifact = self._entries[name]
        result = self._reader(job_id, artifact.relative_path)
        if (
            result.artifact != artifact
            or len(result.payload) != artifact.size_bytes
            or hashlib.sha256(result.payload).hexdigest() != artifact.sha256
        ):
            raise ValueError("Studio evidence seed does not match its source record.")
        return result.payload


def prepare_evidence_process_payload(
    inputs: Mapping[str, object], records: Sequence[StudioJobRecord], max_input_bytes: int
) -> dict[str, object]:
    """Validate complete snapshots and their total metadata-plus-artifact bytes.

    No source bytes are read here. Exceeding the explicit aggregate limit raises
    ``EvidenceInputLimitExceeded`` before job admission or seed directory writes.
    Invalid writer inputs or records raise ``ValueError`` without dropping fields.

    Parameters
    ----------
    inputs : mapping
        Every original writer input category, excluding executable callbacks.
    records : sequence of StudioJobRecord
        Complete source snapshots; declared artifact sizes contribute to the cap.
    max_input_bytes : int
        Positive aggregate byte ceiling for encoded metadata and all seed copies.

    Returns
    -------
    dict[str, object]
        Validated JSON-compatible envelope for the named process task.

    Raises
    ------
    EvidenceInputLimitExceeded
        Aggregate metadata and declared binary bytes exceed the configured cap.
    ValueError
        The limit, JSON input envelope or declared sizes are invalid.
    """
    limit = validate_evidence_input_limit(max_input_bytes)
    payload: dict[str, object] = {
        "inputs": dict(inputs),
        "records": [record.to_public_dict() for record in records],
        "max_input_bytes": limit,
    }
    _Envelope.model_validate(payload)
    _check_budget(payload, records, limit)
    return payload


def execute_evidence_bundle_task(
    context: StudioJobContext, payload: Mapping[str, object]
) -> dict[str, object]:
    """Recheck snapshots/seeds and invoke the original complete bundle writer.

    Parameters
    ----------
    context : StudioJobContext
        Registered worker's bounded seed and output context.
    payload : mapping
        Exact inputs, complete records and API-selected max_input_bytes.

    Returns
    -------
    dict[str, object]
        Original bundle receipt; the existing writer owns all evidence semantics.

    Raises
    ------
    ValueError
        Envelope, record, byte budget, seed integrity or evidence validation fails.
    """
    envelope = _Envelope.model_validate(dict(payload))
    limit = validate_evidence_input_limit(envelope.max_input_bytes)
    records = tuple(decode_job_snapshot(record) for record in envelope.records)
    _check_budget(payload, records, limit)
    seeds = {
        (job_id, artifact.relative_path): (name, artifact)
        for name, job_id, artifact in _seed_entries(records)
    }

    def reader(job_id: str, path: str) -> StudioJobArtifactPayload:
        name, artifact = seeds[(job_id, path)]
        return StudioJobArtifactPayload(artifact=artifact, payload=context.read_seed_input(name))

    inputs = envelope.inputs
    return dict(
        write_studio_evidence_bundle(
            context,
            project_payload=inputs.project_payload,
            simulation_payloads=inputs.simulation_payloads,
            analysis_payloads=inputs.analysis_payloads,
            model_scan_payloads=inputs.model_scan_payloads,
            weight_restore_payloads=inputs.weight_restore_payloads,
            weight_restore_attach_payloads=inputs.weight_restore_attach_payloads,
            default_flow_runs=inputs.default_flow_runs,
            default_flow_attestations=inputs.default_flow_attestations,
            audit_export=inputs.audit_export,
            command_replay=inputs.command_replay,
            job_records=records,
            artifact_reader=reader,
        ).to_public_dict()
    )
