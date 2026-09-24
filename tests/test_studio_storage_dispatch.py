# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — storage authority operation dispatch

"""Every operation reaches its handler through the service's one dispatch.

The API side uses the production clients over real socket pairs; the service
side reads the first frame as the listener does and hands it to
:func:`serve_operation` with a real ledger, gateway and admission. Named
admission stages its seeds in a real held authority directory.
"""

from __future__ import annotations

from collections.abc import Iterator
import os
from pathlib import Path
import socket
import threading
import time

import pytest

from sc_neurocore.studio.platform.jobs_admission import StudioJobQueueFull
from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_schema import StudioJobSubmission
from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.jobs_shared_admission import SharedJobAdmission
from sc_neurocore.studio.platform.policy_gateway import PolicyGateway
from sc_neurocore.studio.platform.policy_models import InMemoryAuditSink
from sc_neurocore.studio.platform.storage_admission_client import (
    read_named_admission_result,
    send_named_admission_request,
)
from sc_neurocore.studio.platform.storage_admission_protocol import StorageNamedAdmissionRequest
from sc_neurocore.studio.platform.storage_configuration import StorageBoundaryConfiguration
from sc_neurocore.studio.platform.storage_dispatch import (
    NamedAdmissionHandler,
    StorageServices,
    serve_operation,
)
from sc_neurocore.studio.platform.storage_named_admission import PreparedNamedAdmission
from sc_neurocore.studio.platform.storage_peer import read_verified_frame, write_verified_frame
from sc_neurocore.studio.platform.storage_query_client import exchange_query, query_request
from sc_neurocore.studio.platform.storage_record_client import read_storage_record
from sc_neurocore.studio.platform.storage_record_protocol import (
    StorageRecordRequest,
    StorageRequester,
)

ADMIN = StorageRequester(principal_id="operator", roles=("studio.admin",))
FRAME = 4096
JOB = "sj_" + "5" * 16


@pytest.fixture
def ledger(tmp_path: Path) -> Iterator[StudioJobLedger]:
    authority = StudioJobLedger(root=tmp_path / "authority", supervisor="storage:1:1")
    (tmp_path / "authority").chmod(0o700)
    try:
        yield authority
    finally:
        authority.close()


def _services(
    ledger: StudioJobLedger,
    *,
    admit_named: NamedAdmissionHandler | None = None,
    authority_dirfd: int | None = None,
) -> StorageServices:
    return StorageServices(
        ledger=ledger,
        gateway=PolicyGateway(InMemoryAuditSink()),
        workspace="default",
        api_uid=os.getuid(),
        frame_max_bytes=FRAME,
        max_metadata_bytes=FRAME,
        max_seed_bytes=8,
        max_seed_entries=2,
        max_manifest_bytes=64,
        max_artifact_bytes=65536,
        max_artifact_entries=16,
        admit_named=admit_named,
        authority_dirfd=authority_dirfd,
    )


def _serving(
    services: StorageServices,
) -> tuple[socket.socket, list[BaseException], threading.Thread]:
    """Serve one connection as the listener does; return the API end."""
    client, service = socket.socketpair()
    failures: list[BaseException] = []

    def serve() -> None:
        deadline = time.monotonic() + 10
        try:
            with service:
                frame = read_verified_frame(
                    service, expected_uid=os.getuid(), max_bytes=FRAME, deadline=deadline
                )
                serve_operation(service, frame, services, deadline=deadline)
        except BaseException as exc:
            failures.append(exc)
        finally:
            services.ledger.close()

    thread = threading.Thread(target=serve)
    thread.start()
    return client, failures, thread


def _admit(ledger: StudioJobLedger) -> None:
    SharedJobAdmission(ledger, max_concurrent=2, max_queued=0).admit(
        job_id=JOB,
        kind="analysis",
        actor="operator",
        workspace="default",
        request_id=None,
        idempotency_key=None,
        experiment_sha256=None,
        admission=None,
        execution_model="process",
        supervisor=supervisor_identity(),
    )


def test_a_record_read_is_dispatched_to_the_record_handler(ledger: StudioJobLedger) -> None:
    """The record operation answers with the complete snapshot."""
    _admit(ledger)
    client, failures, thread = _serving(_services(ledger))
    request = StorageRecordRequest(
        schema_version="studio.storage.record.v2",
        operation="record",
        request_id="trace",
        job_id=JOB,
        workspace="default",
        requester=ADMIN,
    )
    with client:
        record = read_storage_record(
            client,
            request=request,
            expected_service_uid=os.getuid(),
            max_bytes=FRAME,
            deadline=time.monotonic() + 10,
        )
    thread.join(timeout=10)
    assert failures == [] and record == ledger.record(JOB)


def test_operations_without_their_collaborator_are_refused(ledger: StudioJobLedger) -> None:
    """A query without admission and a named admission without custody are refused."""
    client, failures, thread = _serving(_services(ledger))
    with pytest.raises(EOFError):
        exchange_query(
            client,
            query_request("default", "status", requester=None),
            expected_service_uid=os.getuid(),
            max_bytes=FRAME,
            deadline=time.monotonic() + 10,
        )
    thread.join(timeout=10)
    assert [str(failure) for failure in failures] == ["storage query has no admission"]
    client, failures, thread = _serving(_services(ledger))
    with client:
        write_verified_frame(
            client,
            b'{"schema_version":"studio.storage.admission.v1","operation":"admit_named"}',
            expected_uid=os.getuid(),
            max_bytes=FRAME,
            deadline=time.monotonic() + 10,
        )
        client.settimeout(10)
        assert client.recv(1) == b""
    thread.join(timeout=10)
    assert [type(failure) for failure in failures] == [PermissionError]


def test_named_admission_is_dispatched_with_staged_seeds(
    ledger: StudioJobLedger, tmp_path: Path
) -> None:
    """The production sender, the dispatch and the custody handler admit one job."""
    admission = SharedJobAdmission(ledger, max_concurrent=2, max_queued=0)
    staged: list[tuple[str, int]] = []

    def admit(prepared: PreparedNamedAdmission) -> StudioJobSubmission | StudioJobQueueFull:
        assert prepared.seed_files is not None
        staged.extend((seed.name, seed.size) for seed in prepared.seed_files.files)
        return admission.admit(
            job_id=JOB,
            kind=prepared.task.kind,
            actor=prepared.task.owner,
            workspace="default",
            request_id=None,
            idempotency_key=None,
            experiment_sha256=None,
            admission={},
            execution_model="process",
            replay=prepared.replay,
            supervisor=prepared.supervisor,
        )

    directory = os.open(tmp_path / "authority", os.O_RDONLY | os.O_DIRECTORY)
    services = _services(ledger, admit_named=admit, authority_dirfd=directory)
    configuration = StorageBoundaryConfiguration(
        storage_uid=os.getuid(),
        api_uid=os.getuid() + 1,
        worker_uid=os.getuid() + 2,
        authority_root=tmp_path / "authority",
        spool_root=tmp_path / "spool",
        socket_path=tmp_path / "endpoint" / "storage.sock",
        workspace="default",
        frame_max_bytes=FRAME,
        max_metadata_bytes=FRAME,
        max_seed_bytes=8,
        max_seed_entries=2,
        max_manifest_bytes=64,
        max_artifact_bytes=65536,
        max_artifact_entries=16,
        transfer_timeout_seconds=10.0,
        max_connections=1,
    )
    request = StorageNamedAdmissionRequest(
        schema_version="studio.storage.admission.v1",
        operation="admit_named",
        request_id="trace",
        mutation_id="retry",
        workspace="default",
        requester=ADMIN,
        task_name="analysis.run",
        authorized_route="/api/analysis/jobs",
        payload={"model": "lif"},
        seed_manifest={"input/data.bin": 2},
        execution_timeout_seconds=30.0,
        queue_wait_seconds=None,
        admission=None,
        training_config=None,
        experiment_sha256=None,
    )
    try:
        client, failures, thread = _serving(services)
        with client:
            pending = send_named_admission_request(
                client,
                request=request,
                seed_inputs={"input/data.bin": b"ab"},
                configuration=configuration,
            )
            assert read_named_admission_result(client, pending=pending) == JOB
        thread.join(timeout=10)
    finally:
        os.close(directory)
    assert failures == []
    assert staged == [("input/data.bin", 2)]
    assert ledger.record(JOB).status == "pending"
