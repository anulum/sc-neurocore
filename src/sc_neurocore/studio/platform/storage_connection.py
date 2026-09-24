# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — verified storage client connection

"""Connect the trusted API to an existing service-owned Unix endpoint."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator
import os
import socket
import stat
import sys
from threading import TIMEOUT_MAX

from sc_neurocore.studio.platform.storage_configuration import StorageBoundaryConfiguration
from sc_neurocore.studio.platform.storage_peer import require_storage_peer


@contextmanager
def _endpoint_parent(configuration: StorageBoundaryConfiguration) -> Iterator[int]:
    """Hold the canonical endpoint parent without following any path component."""
    flags = os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open("/", flags)
    try:
        for component in configuration.socket_path.parent.parts[1:]:
            ancestor = os.fstat(descriptor)
            if ancestor.st_uid not in (0, configuration.storage_uid):
                raise PermissionError("storage endpoint ancestor has an untrusted owner")
            if ancestor.st_mode & 0o022 and not ancestor.st_mode & stat.S_ISVTX:
                raise PermissionError("storage endpoint ancestor permits namespace writes")
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        parent = os.fstat(descriptor)
        named = os.stat(configuration.socket_path.parent, follow_symlinks=False)
        if (parent.st_dev, parent.st_ino) != (named.st_dev, named.st_ino):
            raise PermissionError("storage endpoint parent changed")
        if (
            parent.st_uid != configuration.storage_uid
            or parent.st_mode & 0o027
            or not parent.st_mode & 0o010
        ):
            raise PermissionError("storage endpoint parent permissions refuse")
        yield descriptor
    finally:
        os.close(descriptor)


def _endpoint_identity(
    configuration: StorageBoundaryConfiguration, directory: int
) -> tuple[int, int]:
    """Require an unchanged service-owned, group-visible socket inode."""
    parent = os.fstat(directory)
    held = os.stat(configuration.socket_path.name, dir_fd=directory, follow_symlinks=False)
    named = os.stat(configuration.socket_path, follow_symlinks=False)
    if (
        not stat.S_ISSOCK(held.st_mode)
        or (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino)
        or held.st_uid != configuration.storage_uid
        or held.st_gid != parent.st_gid
        or stat.S_IMODE(held.st_mode) != 0o660
    ):
        raise PermissionError("storage endpoint identity refuses")
    return held.st_dev, held.st_ino


def connect_storage_authority(configuration: StorageBoundaryConfiguration) -> socket.socket:
    """Return one peer-verified stream from the configured API identity.

    The caller exclusively owns and closes the returned connection. This
    confirms endpoint and kernel peer identity before any frame is sent; it
    does not authenticate a browser principal or enable the isolated runtime.
    No path, credential or UID is accepted from a worker request.
    """
    if sys.platform != "linux" or os.getresuid() != (
        configuration.api_uid,
        configuration.api_uid,
        configuration.api_uid,
    ):
        raise PermissionError("storage client requires the configured Linux API identity")
    if len(os.fsencode(str(configuration.socket_path))) >= 108:
        raise ValueError("storage socket path exceeds the Unix endpoint limit")
    if configuration.transfer_timeout_seconds > TIMEOUT_MAX:
        raise ValueError("storage transfer timeout exceeds the platform limit")
    with _endpoint_parent(configuration) as directory:
        identity = _endpoint_identity(configuration, directory)
        channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            channel.settimeout(configuration.transfer_timeout_seconds)
            channel.connect(f"/proc/self/fd/{directory}/{configuration.socket_path.name}")
            require_storage_peer(channel, expected_uid=configuration.storage_uid)
            if _endpoint_identity(configuration, directory) != identity:
                raise PermissionError("storage endpoint changed during connection")
            return channel
        except BaseException:
            channel.close()
            raise
