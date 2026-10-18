# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Identity file custody and writer serialisation

"""Read identity descriptors once and serialise complete store mutations."""

from __future__ import annotations

import logging
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sc_neurocore.studio.platform.identity_refusals import StudioIdentityRefused
from sc_neurocore.studio.workspace_lock import workspace_lock

_LOGGER = logging.getLogger("sc_neurocore.studio.identity")


def private_identity_path(path: Path) -> Path:
    """Resolve the parent and refuse a writable parent or a symbolic-link store.

    POSIX stores must belong to the running account in a directory it owns
    which group and other accounts cannot write. Windows uses its filesystem
    ACLs. Parent aliases resolve to the same store and writer-lock location.
    """
    path = path.expanduser()
    parent = path.parent.resolve(strict=True)
    info = parent.stat()
    if not stat.S_ISDIR(info.st_mode):
        raise StudioIdentityRefused("Studio identity parent path must be a directory.")
    if os.name == "posix" and (info.st_uid != os.geteuid() or info.st_mode & 0o022):
        raise StudioIdentityRefused(
            "Studio identity parent must be owned by the running account and not writable "
            "by group or other accounts."
        )
    canonical = parent / path.name
    if canonical.is_symlink():
        raise StudioIdentityRefused("Studio identity file must not be a symbolic link.")
    return canonical


def read_identity_text(path: Path) -> str:
    """Read one no-follow descriptor after checking its owner and permissions.

    Owner-readable copies retain the existing migration behaviour: permissions
    are narrowed on the opened descriptor and a warning records that action.
    No subsequent path open can substitute a different file for the checked one.
    """
    canonical = private_identity_path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(canonical, flags)
    with os.fdopen(fd, "r", encoding="utf-8") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise StudioIdentityRefused("Studio identity file must be a regular file.")
        if os.name == "posix":
            if info.st_uid != os.geteuid():
                raise StudioIdentityRefused(
                    "Studio identity file must be owned by the account running Studio."
                )
            if info.st_mode & 0o077:
                os.fchmod(stream.fileno(), 0o600)
                _LOGGER.warning(
                    "Studio identity file %s was readable beyond its owner (mode %o); narrowed to 600",
                    canonical.name,
                    info.st_mode & 0o777,
                )
        return stream.read()


@contextmanager
def identity_store_mutation(path: Path) -> Iterator[Path]:
    """Hold a stable path-based writer lock across load, validation and replacement.

    The lock survives atomic replacements of the JSON file and is shared by
    every CLI and API mutation on this host. Its bounded wait is ten seconds.
    """
    canonical = private_identity_path(path)
    with workspace_lock(canonical.parent, f"identity-{canonical.name}"):
        yield canonical
