# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — worker launcher endpoint

"""Own the launcher's Unix endpoint inside a directory private to the launcher."""

from __future__ import annotations

import os
from pathlib import Path
import socket


class LauncherEndpoint:
    """Bind, and later remove, exactly one launcher socket inode.

    The socket parent must be owned by the launcher and grant at most group
    traversal, so only the configured socket group (the API) can reach it.
    An existing entry is never adopted; shutdown unlinks only the unchanged
    inode this endpoint created.
    """

    def __init__(self, socket_path: Path) -> None:
        """Retain the configured endpoint path; nothing is bound yet."""
        self._socket_path = socket_path
        self._parent_fd: int | None = None
        self._identity: tuple[int, int] | None = None

    def open(self) -> socket.socket:
        """Bind and listen on a new socket through the held parent directory.

        Returns
        -------
        socket.socket
            Listening, non-inheritable socket owned by the caller.

        Raises
        ------
        PermissionError
            The parent is not private to the launcher.
        FileExistsError
            An entry with the endpoint name already exists.
        OSError
            Bind, mode change or listen fails.
        """
        name = self._socket_path.name
        parent = os.open(
            self._socket_path.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        listener: socket.socket | None = None
        identity: tuple[int, int] | None = None
        try:
            held = os.fstat(parent)
            if held.st_uid != os.geteuid() or held.st_mode & 0o027:
                raise PermissionError("launcher socket parent must be private to the launcher")
            try:
                os.stat(name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise FileExistsError("launcher endpoint already exists")
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.set_inheritable(False)
            listener.bind(f"/proc/self/fd/{parent}/{name}")
            # The parent is private to this identity, so nothing else can
            # replace the entry just bound; its inode is recorded for removal.
            created = os.stat(name, dir_fd=parent, follow_symlinks=False)
            identity = (created.st_dev, created.st_ino)
            # The API connects through the socket group; peer UID is checked per request.
            os.chmod(name, 0o660, dir_fd=parent, follow_symlinks=False)  # nosec B103
            listener.listen(8)
        except BaseException:
            if listener is not None:
                listener.close()
            if identity is not None:
                os.unlink(name, dir_fd=parent)
            os.close(parent)
            raise
        self._parent_fd = parent
        self._identity = identity
        return listener

    def close(self) -> None:
        """Remove the endpoint only if it is still the inode this object bound."""
        parent = self._parent_fd
        identity = self._identity
        self._parent_fd = None
        self._identity = None
        if parent is None or identity is None:
            return
        try:
            name = self._socket_path.name
            current = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if (current.st_dev, current.st_ino) == identity:
                os.unlink(name, dir_fd=parent)
        except FileNotFoundError:
            pass
        finally:
            os.close(parent)
