# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Storage namespace descriptor custody

"""Inspect existing service namespaces without creating or repairing storage."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import stat
import sys

from sc_neurocore.studio.platform.storage_configuration import StorageBoundaryConfiguration


@dataclass(frozen=True, slots=True)
class StorageDirectoryDescriptors:
    """Borrowed noninheritable directory handles valid only inside their context.

    Callers must not close, retain or delegate these handles to lower-trust
    processes. Descriptors are ownership evidence for opened objects, not a
    complete mount, ACL, capability or process-isolation qualification.
    """

    authority: int
    endpoint_parent: int


def _check_ancestor(descriptor: int, service_uid: int) -> None:
    metadata = os.fstat(descriptor)
    if metadata.st_uid not in (0, service_uid):
        raise PermissionError("storage ancestor is not owned by a trusted identity")
    if metadata.st_mode & 0o022 and not metadata.st_mode & stat.S_ISVTX:
        raise PermissionError("storage ancestor permits unprotected namespace writes")


@contextmanager
def _open_directory(path: Path, service_uid: int) -> Iterator[int]:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open("/", flags)
    try:
        _check_ancestor(descriptor, service_uid)
        for component in path.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            previous = descriptor
            descriptor = child
            os.close(previous)
            _check_ancestor(descriptor, service_uid)
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def open_storage_directories(
    configuration: StorageBoundaryConfiguration,
) -> Iterator[StorageDirectoryDescriptors]:
    """Hold existing authority and endpoint directories after ownership checks.

    Parameters
    ----------
    configuration : StorageBoundaryConfiguration
        Validated role/path intent. Authority and endpoint parent must exist.

    Yields
    ------
    StorageDirectoryDescriptors
        Borrowed handles, closed on acquisition failure or context exit.

    Raises
    ------
    PermissionError
        Platform, current service IDs, ancestor ownership or final modes refuse.
    OSError
        A directory cannot be opened, including missing paths or symlinks.

    Notes
    -----
    Performs no mkdir, permission repair, ledger opening or listener binding.
    Authority mode must be exactly 0700; endpoint parent cannot admit group or
    other writes. Root/service-owned sticky ancestors are permitted. Privileged
    namespace changes, non-POSIX permissions and inherited capabilities require
    separate deployment checks. Same-UID tests do not establish worker isolation.
    """
    uid = configuration.storage_uid
    if sys.platform != "linux" or os.getresuid() != (uid, uid, uid):
        raise PermissionError("storage namespace requires the configured Linux service identity")
    with _open_directory(configuration.authority_root, uid) as authority:
        metadata = os.fstat(authority)
        if metadata.st_uid != uid or stat.S_IMODE(metadata.st_mode) != 0o700:
            raise PermissionError("storage authority requires service-owned mode 0700")
        with _open_directory(configuration.socket_path.parent, uid) as endpoint:
            metadata = os.fstat(endpoint)
            if metadata.st_uid != uid or metadata.st_mode & 0o022 or not metadata.st_mode & 0o100:
                raise PermissionError("storage endpoint parent ownership or permissions refuse")
            yield StorageDirectoryDescriptors(authority=authority, endpoint_parent=endpoint)
