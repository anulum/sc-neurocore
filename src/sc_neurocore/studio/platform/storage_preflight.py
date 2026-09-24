# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — preflight of the isolated Studio API

"""Check, before an isolated API starts, what its boundary relies on.

Every check reads the running process or the host as they are and reports
each failure by name; nothing is repaired. The API must run as the
configured API identity with enforced route policies, no header identity, a
configured identity store and a persistent audit log. The host must protect
hard and symbolic links, because the API reads spool entries a worker
created. The compute spool root must be the API's, closed to others and
traversable by the compute group, and the API must belong to that group to
read worker output; the launcher endpoint's parent must exist. The only
launcher backend spawns workers directly and proves neither descendant
termination nor per-job accounting, so startup also requires the operator to
have accepted those limits explicitly in the launcher settings.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import stat

from sc_neurocore.studio.platform.settings import StudioRuntimeSettings
from sc_neurocore.studio.platform.storage_configuration import StorageBoundaryConfiguration
from sc_neurocore.studio.platform.storage_launcher_client_settings import LauncherClientSettings

_SYSCTL = Path("/proc/sys/fs")


@dataclass(frozen=True, slots=True)
class PreflightFailure:
    """One failed check and why it failed."""

    check: str
    detail: str


def _sysctl(root: Path, name: str) -> str:
    try:
        return (root / name).read_text(encoding="ascii").strip()
    except OSError:
        return "unreadable"


def isolated_preflight(
    settings: StudioRuntimeSettings, *, sysctl_root: Path = _SYSCTL
) -> tuple[PreflightFailure, ...]:
    """Return every failed check for an isolated API with ``settings``.

    Parameters
    ----------
    settings : StudioRuntimeSettings
        The runtime settings the API would start with.
    sysctl_root : Path
        Directory holding the ``fs`` link-protection values; the host's
        ``/proc/sys/fs`` unless a caller inspects another tree.

    Returns
    -------
    tuple of PreflightFailure
        Empty when every check passed.
    """
    failures: list[PreflightFailure] = []

    def require(check: str, passed: bool, detail: str) -> None:
        if not passed:
            failures.append(PreflightFailure(check, detail))

    boundary, launcher = settings.storage_boundary, settings.storage_launcher
    require("boundary", boundary is not None, "no storage boundary is configured")
    require("launcher", launcher is not None, "no launcher client is configured")
    require("policies", settings.enforce_route_policies, "route policies are not enforced")
    require("header_identity", not settings.allow_header_principal, "header identity is allowed")
    require("identity_store", settings.identity_file_path is not None, "no identity store")
    require("audit", settings.audit_log_path is not None, "no persistent audit log")
    for name in ("protected_hardlinks", "protected_symlinks"):
        value = _sysctl(sysctl_root, name)
        require(name, value == "1", f"fs.{name} is {value}")
    if boundary is not None:
        ids = os.getresuid()
        require(
            "identity",
            ids == (boundary.api_uid,) * 3,
            f"process identities {ids} are not the API identity {boundary.api_uid}",
        )
        try:
            spool = os.stat(boundary.spool_root, follow_symlinks=False)
        except OSError as exc:
            require("spool", False, f"spool root is unavailable: {exc.strerror}")
        else:
            require(
                "spool",
                stat.S_ISDIR(spool.st_mode)
                and spool.st_uid == boundary.api_uid
                and not spool.st_mode & 0o002
                and bool(spool.st_mode & 0o010),
                "spool root must be an API-owned directory, closed to others, "
                "traversable by the compute group",
            )
    if launcher is not None:
        require(
            "compute_group",
            launcher.worker_gid in {os.getegid(), *os.getgroups()},
            f"the API is not a member of the compute group {launcher.worker_gid}",
        )
        require(
            "launcher_endpoint",
            launcher.socket_path.parent.is_dir(),
            "launcher endpoint directory does not exist",
        )
        require(
            "launcher_backend",
            launcher.accept_direct_backend_limits,
            "the direct launcher backend does not prove descendant termination or "
            "per-job accounting, and its limits were not explicitly accepted",
        )
    return tuple(failures)


def require_isolated_preflight(
    settings: StudioRuntimeSettings,
) -> tuple[StorageBoundaryConfiguration, LauncherClientSettings]:
    """Refuse to start an isolated API while any check fails.

    Returns
    -------
    tuple
        The checked storage boundary and launcher client settings.

    Raises
    ------
    RuntimeError
        Names every failed check.
    """
    failures = isolated_preflight(settings)
    boundary, launcher = settings.storage_boundary, settings.storage_launcher
    # A missing boundary or launcher is itself a named failure.
    if failures or boundary is None or launcher is None:
        detail = "; ".join(f"{failure.check}: {failure.detail}" for failure in failures)
        raise RuntimeError(f"Studio isolated preflight failed: {detail}")
    return boundary, launcher


__all__ = ["PreflightFailure", "isolated_preflight", "require_isolated_preflight"]
