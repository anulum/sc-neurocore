# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bounded event contract admission transport

"""Transfer event declarations independently of the small admission metadata."""

from __future__ import annotations

import hashlib
import json
import socket
from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from sc_neurocore.studio.platform.storage_admission_protocol import (
    StorageNamedAdmissionRequest,
    decode_named_admission_request,
)
from sc_neurocore.studio.platform.storage_peer import read_verified_frame, write_verified_frame
from sc_neurocore.studio.platform.training_config_storage import (
    EVENT_DATA_MAX_BYTES,
    REFERENCE_KEY,
    prepare_training_config,
    restore_training_config,
)


class EventAdmissionEnvelope(BaseModel):
    """Small admission request bound to one separately transferred event declaration."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)
    schema_version: Literal["studio.storage.admission.v2"]
    operation: Literal["admit_named"]
    request: StorageNamedAdmissionRequest
    event_bytes: Annotated[int, Field(gt=0, le=EVENT_DATA_MAX_BYTES)]
    event_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def compact_event_admission(
    payload: dict[str, object], training_config: dict[str, object] | None
) -> tuple[dict[str, object], dict[str, object] | None, bytes | None]:
    """Separate a validated event contract while preserving all other request fields.

    Parameters
    ----------
    payload, training_config : dict or None
        Full process payload and the corresponding training snapshot.

    Returns
    -------
    tuple
        Small payload, small snapshot and canonical event bytes, when present.

    Raises
    ------
    ValueError
        Configuration, event custody budget or payload/snapshot identity differs.
    """
    if training_config is None or training_config.get("event_data") is None:
        return payload, training_config, None
    small_json, event_json = prepare_training_config(training_config)
    # A non-null event field either validates to an event contract or raises.
    event_json = cast(str, event_json)
    small = json.loads(small_json)
    restored = restore_training_config(small, event_json)
    wrapped = "config" in payload
    if payload.get("config", payload) != restored:
        raise ValueError("event admission snapshot differs from process payload")
    # Keep the original snapshot's explicit/default field distinction.
    snapshot = dict(training_config)
    del snapshot["event_data"]
    snapshot[REFERENCE_KEY] = small[REFERENCE_KEY]
    process = {**payload, "config": small} if wrapped else small
    return process, snapshot, event_json.encode("utf-8")


def encode_event_admission(
    request: StorageNamedAdmissionRequest, *, max_metadata_bytes: int
) -> tuple[bytes, bytes | None]:
    """Freeze a bounded header and the separately bounded event declaration.

    Parameters
    ----------
    request : StorageNamedAdmissionRequest
        Full typed admission before transfer.
    max_metadata_bytes : int
        Unchanged metadata ceiling for the complete header.

    Returns
    -------
    tuple
        Exact metadata frame and optional canonical event content.
    """
    payload, config, events = compact_event_admission(request.payload, request.training_config)
    if events is None:
        metadata = request.model_dump_json().encode("utf-8")
    else:
        compact = request.model_copy(update={"payload": payload, "training_config": config})
        metadata = (
            EventAdmissionEnvelope(
                schema_version="studio.storage.admission.v2",
                operation="admit_named",
                request=compact,
                event_bytes=len(events),
                event_sha256=hashlib.sha256(events).hexdigest(),
            )
            .model_dump_json()
            .encode("utf-8")
        )
    if not 0 < len(metadata) <= max_metadata_bytes:
        raise ValueError("storage admission metadata exceeds byte limit")
    return metadata, events


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON names before the envelope decoder can lose them."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate event admission field")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    """Reject nonfinite JSON extensions before authorization or bulk receipt."""
    raise ValueError("nonfinite event admission constant")


def decode_event_admission(
    metadata: bytes, *, max_metadata_bytes: int
) -> tuple[StorageNamedAdmissionRequest, EventAdmissionEnvelope | None]:
    """Read a small request without receiving content before policy authorization.

    Parameters
    ----------
    metadata : bytes
        Peer-verified bounded first frame.
    max_metadata_bytes : int
        Unchanged complete metadata ceiling.

    Returns
    -------
    tuple
        Small request and optional bounded event transfer declaration.
    """
    if not 0 < len(metadata) <= max_metadata_bytes:
        raise ValueError("storage admission metadata exceeds byte limit")
    parsed = json.loads(metadata, object_pairs_hook=_unique_fields, parse_constant=_reject_constant)
    if isinstance(parsed, dict) and parsed.get("schema_version") == "studio.storage.admission.v2":
        envelope = EventAdmissionEnvelope.model_validate_json(metadata, strict=True)
        return envelope.request, envelope
    return decode_named_admission_request(metadata, max_metadata_bytes=max_metadata_bytes), None


def send_event_admission_content(
    channel: socket.socket,
    content: bytes | None,
    *,
    expected_uid: int,
    frame_max_bytes: int,
    deadline: float,
) -> None:
    """Send admitted event bytes as exact frames under one deadline and peer UID.

    Parameters
    ----------
    channel : socket.socket
        Exclusively owned authority connection.
    content : bytes, optional
        Canonical declaration returned by the admission encoder.
    expected_uid : int
        Configured storage identity.
    frame_max_bytes : int
        Unchanged positive per-frame ceiling.
    deadline : float
        Absolute transfer deadline shared with metadata and seeds.
    """
    if content is not None:
        for offset in range(0, len(content), frame_max_bytes):
            write_verified_frame(
                channel,
                content[offset : offset + frame_max_bytes],
                expected_uid=expected_uid,
                max_bytes=frame_max_bytes,
                deadline=deadline,
            )


def receive_event_admission_content(
    channel: socket.socket,
    request: StorageNamedAdmissionRequest,
    envelope: EventAdmissionEnvelope | None,
    *,
    expected_uid: int,
    frame_max_bytes: int,
    deadline: float,
) -> StorageNamedAdmissionRequest:
    """Verify all declared bytes and restore the request only after authorization.

    Parameters
    ----------
    channel : socket.socket
        Authorized API connection.
    request : StorageNamedAdmissionRequest
        Small request decoded from its first frame.
    envelope : EventAdmissionEnvelope, optional
        Typed declaration bounded by the existing 64 MiB event custody limit.
    expected_uid : int
        Configured API identity.
    frame_max_bytes : int
        Unchanged per-frame ceiling.
    deadline : float
        Same absolute transfer deadline as metadata and seeds.

    Returns
    -------
    StorageNamedAdmissionRequest
        Full request after exact frame lengths, content SHA and references agree.

    Raises
    ------
    ValueError
        Content length, SHA, configuration reference or canonical JSON differs.
    """
    if envelope is None:
        return request
    content = bytearray()
    for offset in range(0, envelope.event_bytes, frame_max_bytes):
        chunk = read_verified_frame(
            channel, expected_uid=expected_uid, max_bytes=frame_max_bytes, deadline=deadline
        )
        if len(chunk) != min(frame_max_bytes, envelope.event_bytes - offset):
            raise ValueError("event admission chunk length differs")
        content.extend(chunk)
    if hashlib.sha256(content).hexdigest() != envelope.event_sha256:
        raise ValueError("event admission SHA differs")
    if request.training_config is None:
        raise ValueError("event admission requires a training snapshot")
    text = content.decode("utf-8")
    config = restore_training_config(request.training_config, text)
    process = request.payload.get("config", request.payload)
    if not isinstance(process, dict):
        raise ValueError("event admission process configuration differs")
    full_process = restore_training_config(process, text)
    payload = (
        {**request.payload, "config": full_process} if "config" in request.payload else full_process
    )
    restored = request.model_copy(update={"payload": payload, "training_config": config})
    # Apply the same payload/snapshot and independent byte checks as the sender.
    compact_event_admission(restored.payload, restored.training_config)
    return restored
