# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — bounded storage seed ingress

"""Receive exact seed bytes from a verified API peer before admission."""

from __future__ import annotations

from collections.abc import Mapping
import math
import socket
from threading import TIMEOUT_MAX
import time

from sc_neurocore.studio.platform.jobs_paths import _relative_path_candidate
from sc_neurocore.studio.platform.storage_peer import (
    read_verified_frame,
    require_storage_peer,
    write_verified_frame,
)


def _validate_deadline(deadline: float) -> None:
    if isinstance(deadline, bool) or not isinstance(deadline, int | float):
        raise ValueError("invalid storage seed deadline")
    try:
        remaining = deadline - time.monotonic()
        if not math.isfinite(deadline) or remaining > TIMEOUT_MAX:
            raise ValueError("invalid storage seed deadline")
    except OverflowError as exc:
        raise ValueError("invalid storage seed deadline") from exc
    if remaining <= 0:
        raise TimeoutError("storage seed deadline expired")


def validate_storage_seed_manifest(
    manifest: Mapping[str, int],
    *,
    frame_max_bytes: int,
    max_seed_bytes: int,
    max_seed_entries: int,
    max_manifest_bytes: int,
    deadline: float,
) -> tuple[str, ...]:
    """Validate one declared seed manifest and return deterministic frame order.

    Parameters
    ----------
    manifest : mapping of str to int
        Canonical relative seed names and their nonnegative byte lengths.
    frame_max_bytes : int
        Positive upper bound for a single nonempty seed frame.
    max_seed_bytes, max_seed_entries, max_manifest_bytes : int
        Aggregate byte, entry and UTF-8 name-byte budgets from trusted settings.
    deadline : float
        One absolute monotonic transfer deadline.

    Returns
    -------
    tuple[str, ...]
        Validated names sorted in the exact sender/receiver transfer order.

    Raises
    ------
    ValueError
        Manifest, path, size, limits or deadline shape is invalid.
    TimeoutError
        The deadline has already expired.

    Notes
    -----
    The service and API client share this path, byte and resource contract.
    Validation does not read a socket or grant admission authority.
    """
    if (
        not isinstance(manifest, Mapping)
        or type(frame_max_bytes) is not int
        or not 0 < frame_max_bytes <= 0xFFFFFFFF
        or type(max_seed_bytes) is not int
        or max_seed_bytes < 0
        or type(max_seed_entries) is not int
        or max_seed_entries < 0
        or type(max_manifest_bytes) is not int
        or max_manifest_bytes <= 0
    ):
        raise ValueError("invalid storage seed ingress limits")
    _validate_deadline(deadline)
    if len(manifest) > max_seed_entries:
        raise ValueError("storage seed manifest exceeds entry limit")
    total_bytes = 0
    manifest_bytes = 0
    for name, size in manifest.items():
        if not isinstance(name, str) or not name or not name.isprintable():
            raise ValueError("invalid storage seed name")
        candidate = _relative_path_candidate(name, error_message="invalid storage seed path")
        if candidate.as_posix() != name:
            raise ValueError("storage seed path must be canonical")
        manifest_bytes += len(name.encode("utf-8"))
        if manifest_bytes > max_manifest_bytes:
            raise ValueError("storage seed manifest exceeds byte limit")
        if type(size) is not int or size < 0:
            raise ValueError("invalid storage seed size")
        total_bytes += size
        if total_bytes > max_seed_bytes:
            raise ValueError("storage seeds exceed aggregate limit")
    return tuple(sorted(manifest))


def receive_storage_seeds(
    channel: socket.socket,
    *,
    manifest: Mapping[str, int],
    expected_api_uid: int,
    frame_max_bytes: int,
    max_seed_bytes: int,
    max_seed_entries: int,
    max_manifest_bytes: int,
    deadline: float,
) -> dict[str, bytes]:
    """Read exactly the declared seeds in sorted-name order on one Unix stream.

    The manifest is metadata already decoded from a bounded request frame.
    Each nonempty seed uses one or more nonempty frames, with no renewed
    deadline. Zero-byte seeds use no frames. The returned bytes are suitable
    for service-derived replay hashing; no claimed checksum is trusted.
    The caller still owns authorization, peer-process custody, the one-request
    connection lifecycle and the worker handoff.

    Parameters
    ----------
    channel : socket.socket
        Exclusively owned connected Unix stream.
    manifest : mapping of str to int
        Exact relative seed names and declared byte lengths from a bounded
        versioned request; no checksum or path is trusted as an authority.
    expected_api_uid : int
        Service-configured OS identity of the trusted API process.
    frame_max_bytes : int
        Positive maximum payload size of each nonempty frame.
    max_seed_bytes, max_seed_entries, max_manifest_bytes : int
        Explicit aggregate seed, entry and UTF-8 name-byte ceilings.
    deadline : float
        Absolute monotonic deadline shared by every frame in this transfer.

    Returns
    -------
    dict[str, bytes]
        Complete received seed content, keyed by its validated relative name.

    Raises
    ------
    ValueError
        Manifest, limits, frame length or deadline is invalid.
    PermissionError
        The connected peer does not have the configured API identity.
    TimeoutError
        The deadline expires, including a zero-byte transfer.
    EOFError
        The peer closes before all declared bytes arrive.
    OSError
        The socket transfer fails. Errors after validation close the stream.
    """
    if type(expected_api_uid) is not int or not 0 <= expected_api_uid < 0xFFFFFFFF:
        raise ValueError("invalid storage API UID")
    if not isinstance(manifest, Mapping):
        raise ValueError("invalid storage seed manifest")
    declarations = dict(manifest.items())
    names = validate_storage_seed_manifest(
        declarations,
        frame_max_bytes=frame_max_bytes,
        max_seed_bytes=max_seed_bytes,
        max_seed_entries=max_seed_entries,
        max_manifest_bytes=max_manifest_bytes,
        deadline=deadline,
    )

    result: dict[str, bytes] = {}
    try:
        require_storage_peer(channel, expected_uid=expected_api_uid)
        for name in names:
            size = declarations[name]
            chunks: list[bytes] = []
            received = 0
            while received < size:
                chunk = read_verified_frame(
                    channel,
                    expected_uid=expected_api_uid,
                    max_bytes=min(frame_max_bytes, size - received),
                    deadline=deadline,
                )
                chunks.append(chunk)
                received += len(chunk)
            result[name] = b"".join(chunks)
        return result
    except BaseException:
        channel.close()
        raise


def send_storage_seeds(
    channel: socket.socket,
    *,
    seed_inputs: Mapping[str, bytes],
    manifest: Mapping[str, int],
    expected_service_uid: int,
    frame_max_bytes: int,
    max_seed_bytes: int,
    max_seed_entries: int,
    max_manifest_bytes: int,
    deadline: float,
) -> None:
    """Send received API seed bytes in the receiver's deterministic order.

    The caller places ``manifest`` in its preceding versioned request frame.
    This function checks it against a snapshot of the exact bytes before any
    seed frame. Invalid content refuses before transfer; an ambiguous transfer
    closes without retry. The caller owns the connection afterward.

    Parameters
    ----------
    channel : socket.socket
        Exclusively owned connected Unix stream.
    seed_inputs : mapping of str to bytes
        Immutable seed bytes from the authenticated API request.
    manifest : mapping of str to int
        Name/size declarations already sent in the bounded request frame.
    expected_service_uid : int
        Trusted configured service OS identity.
    frame_max_bytes : int
        Positive maximum payload size of each nonempty frame.
    max_seed_bytes, max_seed_entries, max_manifest_bytes : int
        Explicit aggregate seed, entry and UTF-8 name-byte ceilings.
    deadline : float
        Absolute monotonic deadline shared by every frame in this transfer.

    Raises
    ------
    ValueError
        Seeds, limits or deadline are invalid before transfer.
    PermissionError
        The connected peer is not the configured service identity.
    TimeoutError
        The absolute transfer deadline expires.
    OSError
        The socket transfer fails. Ambiguous transfers close the stream.
    """
    if not isinstance(seed_inputs, Mapping):
        raise ValueError("storage seed inputs must be named bytes")
    seeds = dict(seed_inputs.items())
    if any(
        not isinstance(name, str) or not isinstance(data, bytes) for name, data in seeds.items()
    ):
        raise ValueError("storage seed inputs must be named bytes")
    if type(expected_service_uid) is not int or not 0 <= expected_service_uid < 0xFFFFFFFF:
        raise ValueError("invalid storage service UID")
    if not isinstance(manifest, Mapping):
        raise ValueError("invalid storage seed manifest")
    declared = dict(manifest.items())
    actual = {name: len(data) for name, data in seeds.items()}
    names = validate_storage_seed_manifest(
        declared,
        frame_max_bytes=frame_max_bytes,
        max_seed_bytes=max_seed_bytes,
        max_seed_entries=max_seed_entries,
        max_manifest_bytes=max_manifest_bytes,
        deadline=deadline,
    )
    if declared != actual:
        raise ValueError("storage seed manifest does not match seed bytes")
    try:
        require_storage_peer(channel, expected_uid=expected_service_uid)
        for name in names:
            data = seeds[name]
            for offset in range(0, len(data), frame_max_bytes):
                write_verified_frame(
                    channel,
                    data[offset : offset + frame_max_bytes],
                    expected_uid=expected_service_uid,
                    max_bytes=frame_max_bytes,
                    deadline=deadline,
                )
    except BaseException:
        channel.close()
        raise
