# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bounded storage view response content

"""Transfer complete snapshot responses independently of their frame ceiling."""

from __future__ import annotations

import hashlib
import json
import socket
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from sc_neurocore.studio.platform.storage_peer import read_verified_frame, write_verified_frame
from sc_neurocore.studio.platform.training_config_storage import EVENT_DATA_MAX_BYTES

ViewSchema = Literal[
    "studio.storage.record.v2",
    "studio.storage.query.v1",
    "studio.storage.cancel.v1",
    "studio.storage.purge.v1",
]


class StorageViewContent(BaseModel):
    """Correlated small header binding one complete storage snapshot response."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: Literal["studio.storage.view-content.v1"]
    content_schema: ViewSchema
    request_id: str | None
    content_bytes: Annotated[int, Field(gt=0, le=0xFFFFFFFF)]
    content_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def view_content_limit(frame_max_bytes: int, maximum: int | None = None) -> int:
    """Validate independent view content and frame limits.

    Parameters
    ----------
    frame_max_bytes : int
        Positive uint32 per-frame ceiling.
    maximum : int, optional
        Trusted operator content ceiling; defaults to the existing event custody
        ceiling plus one frame of controls, within uint32.

    Returns
    -------
    int
        Positive total content limit, independent of the frame ceiling.

    Raises
    ------
    ValueError
        Either supplied limit is not a positive uint32 integer.
    """
    if type(frame_max_bytes) is not int or not 0 < frame_max_bytes <= 0xFFFFFFFF:
        raise ValueError("invalid storage view frame limit")
    if maximum is None:
        return min(EVENT_DATA_MAX_BYTES + frame_max_bytes, 0xFFFFFFFF)
    if type(maximum) is not int or not 0 < maximum <= 0xFFFFFFFF:
        raise ValueError("invalid storage view content limit")
    return maximum


def send_view_content(
    channel: socket.socket,
    content: bytes,
    *,
    content_schema: ViewSchema,
    request_id: str | None,
    expected_uid: int,
    frame_max_bytes: int,
    deadline: float,
    max_content_bytes: int | None = None,
) -> None:
    """Send a complete admitted response with exact chunk boundaries and SHA.

    Parameters
    ----------
    channel : socket.socket
        Peer-verified connection whose read policy already allowed the response.
    content : bytes
        Complete serialized inner record or query response.
    content_schema : str
        Inner response grammar, bound again by the receiver.
    request_id : str, optional
        Correlation from the original read request.
    expected_uid : int
        Configured API peer identity.
    frame_max_bytes : int
        Unchanged individual frame ceiling.
    deadline : float
        Absolute deadline shared by every response frame.
    max_content_bytes : int, optional
        Independent trusted total response ceiling.

    Raises
    ------
    ValueError
        Limits or total content size are invalid, before any response is sent.
    """
    limit = view_content_limit(frame_max_bytes, max_content_bytes)
    if not isinstance(content, bytes) or not 0 < len(content) <= limit:
        raise ValueError("storage view response exceeds content limit")
    if len(content) <= frame_max_bytes:
        write_verified_frame(
            channel,
            content,
            expected_uid=expected_uid,
            max_bytes=frame_max_bytes,
            deadline=deadline,
        )
        return
    header = StorageViewContent(
        schema_version="studio.storage.view-content.v1",
        content_schema=content_schema,
        request_id=request_id,
        content_bytes=len(content),
        content_sha256=hashlib.sha256(content).hexdigest(),
    )
    write_verified_frame(
        channel,
        header.model_dump_json().encode("utf-8"),
        expected_uid=expected_uid,
        max_bytes=frame_max_bytes,
        deadline=deadline,
    )
    for offset in range(0, len(content), frame_max_bytes):
        write_verified_frame(
            channel,
            content[offset : offset + frame_max_bytes],
            expected_uid=expected_uid,
            max_bytes=frame_max_bytes,
            deadline=deadline,
        )


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Refuse duplicate metadata names before classifying a response."""
    fields: dict[str, object] = {}
    for key, value in pairs:
        if key in fields:
            raise ValueError("duplicate storage view field")
        fields[key] = value
    return fields


def _reject_constant(value: str) -> None:
    """Refuse nonfinite metadata extensions."""
    raise ValueError("nonfinite storage view constant")


def read_view_content(
    channel: socket.socket,
    *,
    content_schema: ViewSchema,
    request_id: str | None,
    expected_uid: int,
    frame_max_bytes: int,
    deadline: float,
    max_content_bytes: int | None = None,
) -> bytes:
    """Read an inline or chunked response without weakening its inner grammar.

    Parameters
    ----------
    channel : socket.socket
        Same exclusively owned authority connection as the original request.
    content_schema : str
        Expected inner grammar; a chunk header cannot select another one.
    request_id : str, optional
        Original correlation checked before allocating content.
    expected_uid : int
        Configured storage authority identity, checked on every frame.
    frame_max_bytes : int
        Unchanged positive uint32 frame ceiling.
    deadline : float
        One absolute request/response deadline.
    max_content_bytes : int, optional
        Independent trusted total response ceiling checked before content receipt.

    Returns
    -------
    bytes
        Exact complete response for the owning record/query decoder to validate.

    Raises
    ------
    ValueError
        Metadata, correlation, content size, chunk length or SHA is invalid.
    """
    limit = view_content_limit(frame_max_bytes, max_content_bytes)
    metadata = read_verified_frame(
        channel, expected_uid=expected_uid, max_bytes=frame_max_bytes, deadline=deadline
    )
    if len(metadata) > limit:
        raise ValueError("storage view response exceeds content limit")
    try:
        parsed = json.loads(
            metadata, object_pairs_hook=_unique_fields, parse_constant=_reject_constant
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("invalid storage view metadata JSON") from exc
    if (
        not isinstance(parsed, dict)
        or parsed.get("schema_version") != "studio.storage.view-content.v1"
    ):
        return metadata
    header = StorageViewContent.model_validate_json(metadata, strict=True)
    if header.content_schema != content_schema or header.request_id != request_id:
        raise ValueError("storage view response does not answer the request")
    if header.content_bytes > limit:
        raise ValueError("storage view response exceeds content limit")
    content = bytearray()
    for offset in range(0, header.content_bytes, frame_max_bytes):
        chunk = read_verified_frame(
            channel, expected_uid=expected_uid, max_bytes=frame_max_bytes, deadline=deadline
        )
        if len(chunk) != min(frame_max_bytes, header.content_bytes - offset):
            raise ValueError("storage view chunk length differs")
        content.extend(chunk)
    if hashlib.sha256(content).hexdigest() != header.content_sha256:
        raise ValueError("storage view content SHA differs")
    return bytes(content)
