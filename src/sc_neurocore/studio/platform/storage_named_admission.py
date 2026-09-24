# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — peer-bound named admission preparation

"""Authorize and receive a named submission before launcher custody exists."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import socket

from sc_neurocore.studio.platform.jobs_admission_replay import StorageAdmissionReplay
from sc_neurocore.studio.platform.policy_gateway import PolicyGateway
from sc_neurocore.studio.platform.policy_models import Principal
from sc_neurocore.studio.platform.policy_routes import build_default_studio_route_policy_registry
from sc_neurocore.studio.platform.storage_admission_digest import derive_storage_admission_replay
from sc_neurocore.studio.platform.storage_admission_protocol import (
    decode_named_admission_request,
)
from sc_neurocore.studio.platform.storage_named_tasks import (
    NamedStudioTask,
    resolve_named_studio_task,
)
from sc_neurocore.studio.platform.storage_peer import (
    read_verified_frame,
    require_storage_supervisor_identity,
)
from sc_neurocore.studio.platform.storage_seed_ingress import receive_storage_seeds
from sc_neurocore.studio.platform.storage_seed_files import (
    ReceivedStorageSeedFile,
    ReceivedStorageSeedFiles,
    receive_storage_seed_files,
)


@dataclass(frozen=True, slots=True)
class PreparedNamedAdmission:
    """Authorized service inputs with verified seed content and peer custody.

    This value is not an admitted job. The service must establish a launcher
    handshake and then call its transactional admission owner; disconnects
    before that point have not created a replay outcome or reserved capacity.
    The exact bounded request is retained as bytes so later code cannot mutate
    nested payload or control mappings after replay identity was derived.
    ``seed_files`` owns unlinked service files only until the listener returns
    its response; the admission handler must transfer them before returning.
    ``seed_inputs`` remains the bounded byte path for direct callers.
    """

    request_json: bytes
    task: NamedStudioTask
    requester: Principal
    supervisor: str
    seed_inputs: tuple[tuple[str, bytes], ...]
    replay: StorageAdmissionReplay
    seed_files: ReceivedStorageSeedFiles | None = None


def prepare_named_admission(
    channel: socket.socket,
    *,
    gateway: PolicyGateway,
    workspace: str,
    expected_api_uid: int,
    frame_max_bytes: int,
    max_metadata_bytes: int,
    max_seed_bytes: int,
    max_seed_entries: int,
    max_manifest_bytes: int,
    deadline: float,
    initial_frame: bytes | None = None,
    authority_dirfd: int | None = None,
) -> PreparedNamedAdmission:
    """Read and authorize a real connected API request without ledger mutation.

    Parameters
    ----------
    channel : socket.socket
        Exclusively owned connected Unix stream from the configured API UID.
    gateway : PolicyGateway
        Existing route-policy audit authority; a failed audit refuses.
    workspace : str
        Server-configured scope, never selected by the request.
    expected_api_uid : int
        Trusted configured API OS identity.
    frame_max_bytes, max_metadata_bytes : int
        Framing and complete metadata byte ceilings.
    max_seed_bytes, max_seed_entries, max_manifest_bytes : int
        Explicit aggregate seed, entry and UTF-8 name-byte ceilings.
    deadline : float
        Absolute monotonic deadline shared by metadata and all seed frames.
    initial_frame : bytes or None
        Optional metadata frame already read through this verified channel by
        its listener before operation dispatch. Direct callers leave it unset.
    authority_dirfd : int or None
        Held private service-authority directory for file-backed seed ingress.
        When absent, use the existing bounded byte path for direct callers.

    Returns
    -------
    PreparedNamedAdmission
        Authorized named task, actual seed content, service-derived replay key
        and pidfd-verified API process generation; no job is admitted.

    Raises
    ------
    ValueError
        Request, task, workspace, frame or content contract is invalid.
    PermissionError
        Peer identity or the existing route policy refuses the requester.
    TimeoutError
        The shared transfer deadline expires.
    EOFError
        The peer disconnects before all declared bytes arrive.
    OSError
        A socket transfer fails. Any failure closes the ambiguous stream.
    AuditSinkError
        The policy audit cannot persist its decision.
    """
    staged: ReceivedStorageSeedFiles | None = None
    try:
        if not isinstance(workspace, str) or not workspace:
            raise ValueError("storage workspace must be nonempty")
        supervisor = require_storage_supervisor_identity(channel, expected_uid=expected_api_uid)
        metadata = initial_frame
        if metadata is None:
            metadata = read_verified_frame(
                channel,
                expected_uid=expected_api_uid,
                max_bytes=min(frame_max_bytes, max_metadata_bytes),
                deadline=deadline,
            )
        request = decode_named_admission_request(metadata, max_metadata_bytes=max_metadata_bytes)
        if request.workspace != workspace:
            raise ValueError("storage request does not match configured workspace")
        task = resolve_named_studio_task(
            request.task_name, authorized_route=request.authorized_route
        )
        claim = request.requester
        requester = None if claim is None else Principal(claim.principal_id, frozenset(claim.roles))
        policy = build_default_studio_route_policy_registry().policy_for(
            "POST", request.authorized_route
        )
        decision = gateway.authorize(
            policy,
            principal=requester,
            route=request.authorized_route,
            request_id=request.request_id,
        )
        if not decision.allowed or requester is None:
            raise PermissionError("storage named admission access denied")
        received_bytes: dict[str, bytes] = {}
        seed_inputs: Mapping[str, bytes | ReceivedStorageSeedFile]
        if authority_dirfd is None:
            received_bytes = receive_storage_seeds(
                channel,
                manifest=request.seed_manifest,
                expected_api_uid=expected_api_uid,
                frame_max_bytes=frame_max_bytes,
                max_seed_bytes=max_seed_bytes,
                max_seed_entries=max_seed_entries,
                max_manifest_bytes=max_manifest_bytes,
                deadline=deadline,
            )
            seed_inputs = received_bytes
        else:
            staged = receive_storage_seed_files(
                channel,
                manifest=request.seed_manifest,
                expected_api_uid=expected_api_uid,
                authority_dirfd=authority_dirfd,
                frame_max_bytes=frame_max_bytes,
                max_seed_bytes=max_seed_bytes,
                max_seed_entries=max_seed_entries,
                max_manifest_bytes=max_manifest_bytes,
                deadline=deadline,
            )
            seed_inputs = {seed.name: seed for seed in staged.files}
        require_storage_supervisor_identity(channel, expected_uid=expected_api_uid)
        payload_json = json.dumps(
            request.payload, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        replay = derive_storage_admission_replay(
            requester=requester,
            mutation_id=request.mutation_id,
            workspace=workspace,
            task=task,
            authorized_route=request.authorized_route,
            payload_json=payload_json,
            seed_inputs=seed_inputs,
            execution_timeout_seconds=request.execution_timeout_seconds,
            queue_wait_seconds=request.queue_wait_seconds,
            admission=request.admission,
            training_config=request.training_config,
            experiment_sha256=request.experiment_sha256,
            max_metadata_bytes=max_metadata_bytes,
            max_seed_bytes=max_seed_bytes,
            max_seed_entries=max_seed_entries,
        )
        return PreparedNamedAdmission(
            request_json=metadata,
            task=task,
            requester=requester,
            supervisor=supervisor,
            seed_inputs=tuple(sorted(received_bytes.items())),
            replay=replay,
            seed_files=staged,
        )
    except BaseException:
        if staged is not None:
            staged.close()
        channel.close()
        raise
