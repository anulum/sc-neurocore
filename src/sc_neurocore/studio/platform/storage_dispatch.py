# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — storage authority operation dispatch

"""Route one peer-verified first frame to the handler of its exact operation.

The owning listener authenticates the connection and reads the first frame;
this module selects the handler by the frame's schema version and operation
and passes that frame on, so every operation is served identically however
the connection was accepted. It holds no socket, path or identity policy of
its own beyond the values the service configured.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import socket
from typing import cast

from sc_neurocore.studio.platform.jobs_admission import StudioJobQueueFull
from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_schema import StudioJobSubmission
from sc_neurocore.studio.platform.jobs_shared_admission import SharedJobAdmission
from sc_neurocore.studio.platform.policy_gateway import PolicyGateway
from sc_neurocore.studio.platform.storage_admission_protocol import decode_named_admission_request
from sc_neurocore.studio.platform.storage_admission_response import encode_named_admission_response
from sc_neurocore.studio.platform.storage_artifact_read import serve_artifact_read
from sc_neurocore.studio.platform.storage_cancel import serve_cancel
from sc_neurocore.studio.platform.storage_finish import serve_finish
from sc_neurocore.studio.platform.storage_named_admission import (
    PreparedNamedAdmission,
    prepare_named_admission,
)
from sc_neurocore.studio.platform.storage_operation import classify_storage_operation
from sc_neurocore.studio.platform.storage_peer import write_verified_frame
from sc_neurocore.studio.platform.storage_purge import AuthorityCustody, serve_purge
from sc_neurocore.studio.platform.storage_query import serve_query
from sc_neurocore.studio.platform.storage_record import serve_record_read
from sc_neurocore.studio.platform.storage_seed_files import ReceivedStorageSeedFiles
from sc_neurocore.studio.platform.storage_supervision import serve_supervision

NamedAdmissionHandler = Callable[[PreparedNamedAdmission], StudioJobSubmission | StudioJobQueueFull]


@dataclass(frozen=True, slots=True)
class StorageServices:
    """The authority's collaborators and the configured limits they serve with.

    ``admission`` serves the status view; ``admit_named`` and
    ``authority_dirfd`` serve named admission. An operation whose collaborator
    is absent is refused before its request is decoded.
    """

    ledger: StudioJobLedger
    gateway: PolicyGateway
    workspace: str
    api_uid: int
    frame_max_bytes: int
    max_metadata_bytes: int
    max_seed_bytes: int
    max_seed_entries: int
    max_manifest_bytes: int
    max_artifact_bytes: int
    max_artifact_entries: int
    admission: SharedJobAdmission | None = None
    admit_named: NamedAdmissionHandler | None = None
    authority_dirfd: int | None = None


def _admit(
    channel: socket.socket, metadata: bytes, services: StorageServices, deadline: float
) -> None:
    handler, directory = services.admit_named, services.authority_dirfd
    if handler is None or directory is None:
        raise PermissionError("storage named admission has no custody handler")
    prepared = prepare_named_admission(
        channel,
        gateway=services.gateway,
        workspace=services.workspace,
        expected_api_uid=services.api_uid,
        frame_max_bytes=services.frame_max_bytes,
        max_metadata_bytes=services.max_metadata_bytes,
        max_seed_bytes=services.max_seed_bytes,
        max_seed_entries=services.max_seed_entries,
        max_manifest_bytes=services.max_manifest_bytes,
        deadline=deadline,
        initial_frame=metadata,
        authority_dirfd=directory,
    )
    # Seeds are always staged when an authority directory is supplied.
    staged = cast(ReceivedStorageSeedFiles, prepared.seed_files)
    with staged:
        request = decode_named_admission_request(
            prepared.request_json, max_metadata_bytes=services.max_metadata_bytes
        )
        response = encode_named_admission_response(
            request=request,
            replay=prepared.replay,
            outcome=handler(prepared),
            max_bytes=services.frame_max_bytes,
        )
        write_verified_frame(
            channel,
            response,
            expected_uid=services.api_uid,
            max_bytes=services.frame_max_bytes,
            deadline=deadline,
        )


def serve_operation(
    channel: socket.socket, metadata: bytes, services: StorageServices, *, deadline: float
) -> None:
    """Serve the operation named by ``metadata`` on ``channel``.

    Parameters
    ----------
    channel : socket.socket
        Connected stream whose peer and first frame the caller verified.
    metadata : bytes
        That first frame.
    services : StorageServices
        The service's collaborators and limits.
    deadline : float
        Absolute monotonic wire deadline.

    Raises
    ------
    ValueError
        The frame names no supported operation or its handler refused it.
    PermissionError
        A collaborator the operation needs is not configured, or a handler's
        peer or policy check refused.
    TimeoutError, EOFError, OSError
        Wire transfer fails.
    """
    operation = classify_storage_operation(metadata)
    uid = services.api_uid
    if operation == "record":
        serve_record_read(
            channel,
            ledger=services.ledger,
            gateway=services.gateway,
            workspace=services.workspace,
            max_bytes=services.frame_max_bytes,
            expected_api_uid=uid,
            deadline=deadline,
            initial_frame=metadata,
        )
    elif operation == "finish":
        serve_finish(
            channel,
            ledger=services.ledger,
            workspace=services.workspace,
            frame_max_bytes=services.frame_max_bytes,
            max_artifact_bytes=services.max_artifact_bytes,
            max_artifact_entries=services.max_artifact_entries,
            expected_api_uid=uid,
            deadline=deadline,
            initial_frame=metadata,
        )
    elif operation == "supervision":
        serve_supervision(
            channel,
            ledger=services.ledger,
            workspace=services.workspace,
            max_bytes=services.frame_max_bytes,
            expected_api_uid=uid,
            deadline=deadline,
            initial_frame=metadata,
        )
    elif operation == "purge":
        serve_purge(
            channel,
            custody=AuthorityCustody(services.ledger),
            gateway=services.gateway,
            workspace=services.workspace,
            max_bytes=services.frame_max_bytes,
            expected_api_uid=uid,
            deadline=deadline,
            initial_frame=metadata,
        )
    elif operation == "artifact":
        serve_artifact_read(
            channel,
            ledger=services.ledger,
            gateway=services.gateway,
            workspace=services.workspace,
            max_bytes=services.frame_max_bytes,
            expected_api_uid=uid,
            deadline=deadline,
            initial_frame=metadata,
        )
    elif operation == "cancel":
        serve_cancel(
            channel,
            ledger=services.ledger,
            gateway=services.gateway,
            workspace=services.workspace,
            max_bytes=services.frame_max_bytes,
            expected_api_uid=uid,
            deadline=deadline,
            initial_frame=metadata,
        )
    elif operation == "query":
        if services.admission is None:
            raise PermissionError("storage query has no admission")
        serve_query(
            channel,
            ledger=services.ledger,
            admission=services.admission,
            gateway=services.gateway,
            workspace=services.workspace,
            max_bytes=services.frame_max_bytes,
            expected_api_uid=uid,
            deadline=deadline,
            initial_frame=metadata,
        )
    else:
        _admit(channel, metadata, services, deadline)


__all__ = ["NamedAdmissionHandler", "StorageServices", "serve_operation"]
