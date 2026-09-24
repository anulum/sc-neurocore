# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — storage listener fixture ownership

"""Build one real local listener namespace for focused Unix socket tests."""

import os
from pathlib import Path

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.policy_gateway import PolicyGateway
from sc_neurocore.studio.platform.policy_models import InMemoryAuditSink
from sc_neurocore.studio.platform.storage_configuration import StorageBoundaryConfiguration


def boundary(
    tmp_path: Path, *, timeout: float = 0.25
) -> tuple[StorageBoundaryConfiguration, StudioJobLedger, PolicyGateway]:
    """Create a service-owned fixture root and a separate existing endpoint parent."""
    authority = tmp_path / "authority"
    endpoint = tmp_path / "endpoint"
    authority.mkdir(mode=0o700)
    endpoint.mkdir(mode=0o2750)
    endpoint.chmod(0o2750)
    ledger = StudioJobLedger(root=authority)
    config = StorageBoundaryConfiguration(
        storage_uid=os.getuid(),
        api_uid=os.getuid() + 1,
        worker_uid=os.getuid() + 2,
        authority_root=authority,
        spool_root=tmp_path / "spool",
        socket_path=endpoint / "storage.sock",
        workspace="default",
        frame_max_bytes=8192,
        max_metadata_bytes=4096,
        max_seed_bytes=8192,
        max_seed_entries=16,
        max_manifest_bytes=1024,
        max_artifact_bytes=65536,
        max_artifact_entries=16,
        transfer_timeout_seconds=timeout,
        max_connections=2,
    )
    return config, ledger, PolicyGateway(InMemoryAuditSink())
