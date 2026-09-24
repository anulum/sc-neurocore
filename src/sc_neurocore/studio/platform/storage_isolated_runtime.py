# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — isolated jobs runtime from explicit configuration

"""Build the isolated jobs facade from explicit, validated operator configuration.

The storage boundary names the identities, roots, workspace and budgets; the
launcher client settings name the launcher endpoint, its identity, the
compute group and the supervision cadence. Nothing here has a default that
could silently select a different endpoint or identity, and building the
facade contacts nothing: every later exchange verifies its peer.
"""

from __future__ import annotations

from functools import partial

from sc_neurocore.studio.platform.storage_configuration import StorageBoundaryConfiguration
from sc_neurocore.studio.platform.storage_connection import connect_storage_authority
from sc_neurocore.studio.platform.storage_generation_exchanges import GenerationRuntime
from sc_neurocore.studio.platform.storage_isolated_jobs import IsolatedJobManager
from sc_neurocore.studio.platform.storage_launcher_client_settings import LauncherClientSettings
from sc_neurocore.studio.platform.storage_live_spool import LiveSpools


def build_isolated_job_manager(
    boundary: StorageBoundaryConfiguration,
    launcher: LauncherClientSettings,
    *,
    allowed_kinds: frozenset[str],
    default_timeout_seconds: float,
    max_artifact_bytes: int,
) -> IsolatedJobManager:
    """Return the isolated facade for this API process.

    Parameters
    ----------
    boundary : StorageBoundaryConfiguration
        Validated identities, roots, workspace and transfer budgets.
    launcher : LauncherClientSettings
        Validated launcher endpoint, identity and supervision cadence.
    allowed_kinds : frozenset[str]
        Job kinds the API admits.
    default_timeout_seconds : float
        Execution timeout for jobs submitted without one.
    max_artifact_bytes : int
        Per-artefact budget handed to each worker, as the embedded setting.
    """
    runtime = GenerationRuntime(
        workspace=boundary.workspace,
        spool_root=boundary.spool_root,
        storage_uid=boundary.storage_uid,
        frame_max_bytes=boundary.frame_max_bytes,
        transfer_timeout_seconds=boundary.transfer_timeout_seconds,
        launcher_socket=launcher.socket_path,
        launcher_uid=launcher.launcher_uid,
        worker_uid=boundary.worker_uid,
        worker_gid=launcher.worker_gid,
        max_artifact_bytes=max_artifact_bytes,
        artifact_total_bytes=boundary.max_artifact_bytes,
        artifact_entries=boundary.max_artifact_entries,
        grant_timeout_seconds=launcher.grant_timeout_seconds,
        heartbeat_seconds=launcher.heartbeat_seconds,
        poll_seconds=launcher.poll_seconds,
        attempts=launcher.attempts,
        connect=partial(connect_storage_authority, boundary),
        live=LiveSpools(retain=launcher.live_retain, max_seed_bytes=max_artifact_bytes),
    )
    return IsolatedJobManager(
        runtime,
        boundary,
        allowed_kinds=allowed_kinds,
        default_timeout_seconds=default_timeout_seconds,
    )


__all__ = ["build_isolated_job_manager"]
