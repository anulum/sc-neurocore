# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — storage authority admission of named process jobs

"""Admit one prepared named submission at the storage authority.

The listener has verified the API peer, applied the route's policy, resolved
the reviewed task and received the exact seeds, deriving the replay identity
from the bytes. This handler then admits a process job in one transaction of
the shared admission: kind and owner are the reviewed task's, the lease and
reservation are delegated to the verified API generation, and an identical
replay returns the job it already admitted. A full queue is answered as a
refusal value, never raised through the listener.
"""

from __future__ import annotations

import secrets

from sc_neurocore.studio.platform.jobs_admission import StudioJobQueueFull
from sc_neurocore.studio.platform.jobs_ledger_schema import StudioJobSubmission
from sc_neurocore.studio.platform.jobs_shared_admission import SharedJobAdmission
from sc_neurocore.studio.platform.storage_admission_protocol import decode_named_admission_request
from sc_neurocore.studio.platform.storage_dispatch import NamedAdmissionHandler
from sc_neurocore.studio.platform.storage_named_admission import PreparedNamedAdmission


def named_process_admission(
    admission: SharedJobAdmission, *, workspace: str
) -> NamedAdmissionHandler:
    """Return the service's handler that admits prepared named process jobs.

    Parameters
    ----------
    admission : SharedJobAdmission
        The service's configured admission over its ledger.
    workspace : str
        The server-bound workspace every admitted job belongs to.
    """

    def admit(prepared: PreparedNamedAdmission) -> StudioJobSubmission | StudioJobQueueFull:
        request = decode_named_admission_request(
            prepared.request_json, max_metadata_bytes=len(prepared.request_json)
        )
        try:
            return admission.admit(
                job_id=f"sj_{secrets.token_hex(8)}",
                kind=prepared.task.kind,
                actor=prepared.task.owner,
                workspace=workspace,
                request_id=request.request_id,
                idempotency_key=None,
                experiment_sha256=request.experiment_sha256,
                admission=request.admission,
                execution_model="process",
                training_config=request.training_config,
                timeout_seconds=request.queue_wait_seconds,
                replay=prepared.replay,
                supervisor=prepared.supervisor,
            )
        except StudioJobQueueFull as full:
            return full

    return admit


__all__ = ["named_process_admission"]
