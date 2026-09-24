# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — bounded named storage admission client

"""Send one complete named intent to a verified authority without claiming admission."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import socket
import time

from sc_neurocore.studio.platform.jobs_admission_replay import StorageAdmissionReplay
from sc_neurocore.studio.platform.policy_models import Principal
from sc_neurocore.studio.platform.storage_admission_digest import derive_storage_admission_replay
from sc_neurocore.studio.platform.storage_admission_protocol import (
    StorageNamedAdmissionRequest,
    decode_named_admission_request,
)
from sc_neurocore.studio.platform.storage_admission_response import (
    decode_named_admission_response,
)
from sc_neurocore.studio.platform.storage_configuration import StorageBoundaryConfiguration
from sc_neurocore.studio.platform.storage_named_tasks import resolve_named_studio_task
from sc_neurocore.studio.platform.storage_peer import read_verified_frame, write_verified_frame
from sc_neurocore.studio.platform.storage_seed_ingress import (
    send_storage_seeds,
    validate_storage_seed_manifest,
)


@dataclass(frozen=True, slots=True)
class PendingNamedAdmission:
    """Immutable exact transfer state for one correlated authority reply.

    The original request object may contain mutable nested JSON; only these
    bytes were sent. The service identity and deadline are also snapshotted so
    a later caller cannot replace them with another configuration.
    """

    request_json: bytes
    replay: StorageAdmissionReplay
    service_uid: int
    response_max_bytes: int
    deadline: float


def send_named_admission_request(
    channel: socket.socket,
    *,
    request: StorageNamedAdmissionRequest,
    seed_inputs: Mapping[str, bytes],
    configuration: StorageBoundaryConfiguration,
) -> PendingNamedAdmission:
    """Send validated metadata and exact seed bytes on one connected Unix stream.

    The trusted API caller must construct the requester only from its
    middleware-authenticated principal. This function checks a frozen metadata
    and seed snapshot before sending, uses configuration-owned limits, and
    closes the channel on every failure. On success the caller retains an
    immutable transfer snapshot and the channel for one correlated response.
    The service independently derives and checks its own content identity.
    Sending does not reserve capacity, admit a job or authorize a worker.

    Parameters
    ----------
    channel : socket.socket
        Exclusively owned connected Unix stream to the configured service.
    request : StorageNamedAdmissionRequest
        Typed named metadata built from trusted API identity and route context.
    seed_inputs : mapping of str to bytes
        Exact immutable seed content corresponding to the request manifest.
    configuration : StorageBoundaryConfiguration
        Trusted workspace, service UID, byte budgets and transfer timeout.

    Returns
    -------
    PendingNamedAdmission
        Original sent metadata, expected content identity, service UID and
        absolute reply deadline. It grants no admission authority.

    Raises
    ------
    ValueError
        Request, workspace, metadata, manifest or seed bytes are invalid.
    PermissionError
        The connected service peer has the wrong kernel UID.
    TimeoutError
        The one absolute transfer deadline expires.
    OSError
        A socket transfer fails; the ambiguous stream is closed.
    """
    try:
        if not isinstance(request, StorageNamedAdmissionRequest):
            raise ValueError("storage admission request must be typed")
        if request.workspace != configuration.workspace:
            raise ValueError("storage request does not match configured workspace")
        if not isinstance(seed_inputs, Mapping):
            raise ValueError("storage seed inputs must be named bytes")
        seeds = dict(seed_inputs.items())
        if any(
            not isinstance(name, str) or not isinstance(data, bytes) for name, data in seeds.items()
        ):
            raise ValueError("storage seed inputs must be named bytes")
        metadata = request.model_dump_json().encode("utf-8")
        decoded = decode_named_admission_request(
            metadata, max_metadata_bytes=configuration.max_metadata_bytes
        )
        deadline = time.monotonic() + configuration.transfer_timeout_seconds
        manifest = dict(decoded.seed_manifest)
        validate_storage_seed_manifest(
            manifest,
            frame_max_bytes=configuration.frame_max_bytes,
            max_seed_bytes=configuration.max_seed_bytes,
            max_seed_entries=configuration.max_seed_entries,
            max_manifest_bytes=configuration.max_manifest_bytes,
            deadline=deadline,
        )
        if manifest != {name: len(data) for name, data in seeds.items()}:
            raise ValueError("storage seed manifest does not match seed bytes")
        claim = decoded.requester
        if claim is None:
            raise ValueError("named storage admission requires an authenticated requester")
        requester = Principal(claim.principal_id, frozenset(claim.roles))
        task = resolve_named_studio_task(
            decoded.task_name, authorized_route=decoded.authorized_route
        )
        replay = derive_storage_admission_replay(
            requester=requester,
            mutation_id=decoded.mutation_id,
            workspace=configuration.workspace,
            task=task,
            authorized_route=decoded.authorized_route,
            payload_json=json.dumps(
                decoded.payload, allow_nan=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8"),
            seed_inputs=seeds,
            execution_timeout_seconds=decoded.execution_timeout_seconds,
            queue_wait_seconds=decoded.queue_wait_seconds,
            admission=decoded.admission,
            training_config=decoded.training_config,
            experiment_sha256=decoded.experiment_sha256,
            max_metadata_bytes=configuration.max_metadata_bytes,
            max_seed_bytes=configuration.max_seed_bytes,
            max_seed_entries=configuration.max_seed_entries,
        )
        write_verified_frame(
            channel,
            metadata,
            expected_uid=configuration.storage_uid,
            max_bytes=configuration.max_metadata_bytes,
            deadline=deadline,
        )
        send_storage_seeds(
            channel,
            seed_inputs=seeds,
            manifest=manifest,
            expected_service_uid=configuration.storage_uid,
            frame_max_bytes=configuration.frame_max_bytes,
            max_seed_bytes=configuration.max_seed_bytes,
            max_seed_entries=configuration.max_seed_entries,
            max_manifest_bytes=configuration.max_manifest_bytes,
            deadline=deadline,
        )
        return PendingNamedAdmission(
            request_json=metadata,
            replay=replay,
            service_uid=configuration.storage_uid,
            response_max_bytes=configuration.frame_max_bytes,
            deadline=deadline,
        )
    except BaseException:
        channel.close()
        raise


def read_named_admission_result(channel: socket.socket, *, pending: PendingNamedAdmission) -> str:
    """Read one peer-verified result for the exact previously sent mutation.

    The original send and this response share one absolute deadline. The
    channel is consumed and closed on success, refusal, timeout or corruption.
    A timeout or disconnect is ambiguous: the API must use the same mutation
    ID and content for a deliberate durable replay, never infer no admission.

    Parameters
    ----------
    channel : socket.socket
        Same exclusively owned Unix stream used for the pending send.
    pending : PendingNamedAdmission
        Immutable snapshot returned by that send, never browser input.

    Returns
    -------
    str
        Correlated admitted job ID; read its complete record separately.

    Raises
    ------
    StudioJobQueueFull
        A correlated capacity refusal returned by the service.
    ValueError
        Pending state, framing, schema or response correlation is invalid.
    PermissionError
        The connected service peer UID differs from the sent snapshot.
    TimeoutError
        The original absolute transfer deadline expires.
    EOFError
        The service disconnects before a complete result arrives.
    OSError
        Socket transfer fails.
    """
    with channel:
        if not isinstance(pending, PendingNamedAdmission):
            raise ValueError("invalid pending storage admission")
        request = decode_named_admission_request(
            pending.request_json, max_metadata_bytes=len(pending.request_json)
        )
        payload = read_verified_frame(
            channel,
            expected_uid=pending.service_uid,
            max_bytes=pending.response_max_bytes,
            deadline=pending.deadline,
        )
        return decode_named_admission_response(
            payload,
            request=request,
            replay=pending.replay,
            max_bytes=pending.response_max_bytes,
        )
