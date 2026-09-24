# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — bounded storage record listener

"""Own one peer-verified Unix endpoint without adopting existing storage."""

from __future__ import annotations

from contextlib import ExitStack
import os
import socket
import stat
from threading import TIMEOUT_MAX
import time
from types import TracebackType

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.policy_gateway import PolicyGateway
from sc_neurocore.studio.platform.storage_configuration import StorageBoundaryConfiguration
from sc_neurocore.studio.platform.storage_dispatch import (
    NamedAdmissionHandler,
    StorageServices,
    serve_operation,
)
from sc_neurocore.studio.platform.jobs_shared_admission import SharedJobAdmission
from sc_neurocore.studio.platform.storage_namespace import (
    StorageDirectoryDescriptors,
    open_storage_directories,
)
from sc_neurocore.studio.platform.storage_peer import (
    read_verified_frame,
    require_storage_supervisor_identity,
)


class StorageRecordListener:
    """Serve one bounded versioned request at a time from the configured API UID.

    Startup inspects an existing service-owned namespace and an existing ledger.
    It never creates a database, adopts a stale socket, repairs permissions, or
    starts a worker. Named admission requires an explicitly supplied trusted
    handler that establishes worker custody before returning a durable outcome.
    A caller owns the service loop and calls :meth:`serve_once` repeatedly;
    this class does not claim a deployed isolated profile.
    """

    def __init__(
        self,
        configuration: StorageBoundaryConfiguration,
        *,
        ledger: StudioJobLedger,
        gateway: PolicyGateway,
        admit_named: NamedAdmissionHandler | None = None,
        admission: SharedJobAdmission | None = None,
    ) -> None:
        """Retain explicit configuration and existing authority collaborators."""
        self._configuration = configuration
        self._ledger = ledger
        self._gateway = gateway
        self._admit_named = admit_named
        self._admission = admission
        self._stack: ExitStack | None = None
        self._directories: StorageDirectoryDescriptors | None = None
        self._listener: socket.socket | None = None
        self._socket_identity: tuple[int, int] | None = None

    def _parent_metadata(self, directory: int) -> os.stat_result:
        opened = os.fstat(directory)
        named = os.stat(self._configuration.socket_path.parent, follow_symlinks=False)
        if (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino):
            raise PermissionError("storage socket parent changed")
        return opened

    def _socket_metadata(self, directory: int) -> os.stat_result:
        return os.stat(
            self._configuration.socket_path.name,
            dir_fd=directory,
            follow_symlinks=False,
        )

    def _assert_ledger_identity(self, authority: os.stat_result) -> None:
        try:
            database = os.stat(self._ledger.path, follow_symlinks=False)
        except OSError as exc:
            raise PermissionError("storage ledger is unavailable") from exc
        identity = (authority.st_dev, authority.st_ino, database.st_dev, database.st_ino)
        if (
            not stat.S_ISREG(database.st_mode)
            or database.st_uid != self._configuration.storage_uid
            or identity != self._ledger.storage_identity
        ):
            raise PermissionError("storage ledger identity changed")

    def _assert_endpoint(self) -> None:
        directories = self._directories
        identity = self._socket_identity
        if directories is None or identity is None:
            raise RuntimeError("storage listener is not started")
        try:
            authority = os.fstat(directories.authority)
            canonical_authority = os.stat(self._configuration.authority_root, follow_symlinks=False)
            parent = self._parent_metadata(directories.endpoint_parent)
            named = self._socket_metadata(directories.endpoint_parent)
            canonical = os.stat(self._configuration.socket_path, follow_symlinks=False)
        except OSError as exc:
            raise PermissionError("storage socket endpoint is unavailable") from exc
        if (
            (authority.st_dev, authority.st_ino)
            != (canonical_authority.st_dev, canonical_authority.st_ino)
            or authority.st_uid != self._configuration.storage_uid
            or stat.S_IMODE(authority.st_mode) != 0o700
        ):
            raise PermissionError("storage authority changed")
        self._assert_ledger_identity(authority)
        if (
            parent.st_uid != self._configuration.storage_uid
            or parent.st_mode & 0o027
            or not parent.st_mode & 0o010
        ):
            raise PermissionError("storage socket parent permissions changed")
        if (
            not stat.S_ISSOCK(named.st_mode)
            or (named.st_dev, named.st_ino) != identity
            or (canonical.st_dev, canonical.st_ino) != identity
            or named.st_uid != self._configuration.storage_uid
            or named.st_gid != parent.st_gid
            or stat.S_IMODE(named.st_mode) != 0o660
        ):
            raise PermissionError("storage socket endpoint changed")

    def start(self) -> None:
        """Bind a new socket through held descriptors after strict path checks.

        Raises
        ------
        PermissionError
            Service identity, ledger root, endpoint ownership or mode is unsafe.
        FileExistsError
            The endpoint already exists, including a stale or occupied socket.
        ValueError
            The endpoint path or transfer timeout cannot be used safely.
        OSError
            Descriptor acquisition, bind, mode change or listen fails.
        """
        if self._listener is not None:
            raise RuntimeError("storage listener is already started")
        config = self._configuration
        if len(os.fsencode(str(config.socket_path))) >= 108:
            raise ValueError("storage socket path exceeds the Unix endpoint limit")
        if config.transfer_timeout_seconds > TIMEOUT_MAX:
            raise ValueError("storage transfer timeout exceeds the platform limit")
        stack = ExitStack()
        listener: socket.socket | None = None
        identity: tuple[int, int] | None = None
        try:
            directories = stack.enter_context(open_storage_directories(config))
            parent = self._parent_metadata(directories.endpoint_parent)
            if parent.st_mode & 0o027 or not parent.st_mode & 0o010:
                raise PermissionError("storage socket parent must grant group traversal only")
            if self._ledger.path.parent != config.authority_root:
                raise PermissionError("storage ledger is outside the configured authority")
            ledger_root = os.stat(self._ledger.path.parent, follow_symlinks=False)
            authority = os.fstat(directories.authority)
            if (ledger_root.st_dev, ledger_root.st_ino) != (
                authority.st_dev,
                authority.st_ino,
            ):
                raise PermissionError("storage ledger authority changed")
            self._assert_ledger_identity(authority)
            try:
                self._socket_metadata(directories.endpoint_parent)
            except FileNotFoundError:
                pass
            else:
                raise FileExistsError("storage socket endpoint already exists")

            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.set_inheritable(False)
            listener.bind(f"/proc/self/fd/{directories.endpoint_parent}/{config.socket_path.name}")
            created = self._socket_metadata(directories.endpoint_parent)
            if not stat.S_ISSOCK(created.st_mode) or created.st_uid != config.storage_uid:
                raise PermissionError("storage socket creation identity changed")
            identity = (created.st_dev, created.st_ino)
            if created.st_gid != parent.st_gid:
                raise PermissionError("storage socket group differs from its parent")
            # The API connects through the parent's group; peer UID and pidfd
            # generation are verified before any frame is read.
            os.chmod(  # nosec B103
                config.socket_path.name,
                0o660,
                dir_fd=directories.endpoint_parent,
                follow_symlinks=False,
            )
            listener.listen(config.max_connections)
            listener.settimeout(config.transfer_timeout_seconds)
            self._directories = directories
            self._socket_identity = identity
            self._assert_endpoint()
            self._listener = listener
            self._stack = stack
        except BaseException:
            if listener is not None:
                listener.close()
            if identity is not None:
                try:
                    current = self._socket_metadata(directories.endpoint_parent)
                    if (current.st_dev, current.st_ino) == identity:
                        os.unlink(config.socket_path.name, dir_fd=directories.endpoint_parent)
                except FileNotFoundError:
                    pass
            self._directories = None
            self._socket_identity = None
            stack.close()
            raise

    def serve_once(self) -> None:
        """Accept and serve one request under finite accept and wire deadlines.

        Raises
        ------
        RuntimeError
            The listener has not started.
        TimeoutError
            No client arrives within the configured transfer timeout.
        PermissionError
            Endpoint substitution, peer identity or route policy refuses.
        ValueError
            A workspace mismatch, malformed frame or response limit refuses.

        Notes
        -----
        The accept timeout and subsequent wire deadline are separate. Policy
        audit and SQLite work retain their own existing execution bounds.
        """
        listener, directories = self._listener, self._directories
        # Both are set together by start() and cleared together by stop().
        if listener is None or directories is None:
            raise RuntimeError("storage listener is not started")
        self._assert_endpoint()
        channel, _ = listener.accept()
        with channel:
            self._assert_endpoint()
            config = self._configuration
            deadline = time.monotonic() + config.transfer_timeout_seconds
            require_storage_supervisor_identity(channel, expected_uid=config.api_uid)
            metadata = read_verified_frame(
                channel,
                expected_uid=config.api_uid,
                max_bytes=config.frame_max_bytes,
                deadline=deadline,
            )
            services = StorageServices(
                ledger=self._ledger,
                gateway=self._gateway,
                workspace=config.workspace,
                api_uid=config.api_uid,
                frame_max_bytes=config.frame_max_bytes,
                max_metadata_bytes=config.max_metadata_bytes,
                max_seed_bytes=config.max_seed_bytes,
                max_seed_entries=config.max_seed_entries,
                max_manifest_bytes=config.max_manifest_bytes,
                max_artifact_bytes=config.max_artifact_bytes,
                max_artifact_entries=config.max_artifact_entries,
                admission=self._admission,
                admit_named=self._admit_named,
                authority_dirfd=directories.authority,
            )
            serve_operation(channel, metadata, services, deadline=deadline)

    def stop(self) -> None:
        """Close the listener and remove only its own unchanged socket inode.

        Raises
        ------
        PermissionError
            An endpoint was removed or substituted; it is never unlinked here.
        """
        listener = self._listener
        stack = self._stack
        directories = self._directories
        identity = self._socket_identity
        self._listener = None
        self._stack = None
        self._directories = None
        self._socket_identity = None
        if listener is None or stack is None or directories is None or identity is None:
            return
        listener.close()
        try:
            try:
                current = self._socket_metadata(directories.endpoint_parent)
            except FileNotFoundError as exc:
                raise PermissionError("storage socket endpoint disappeared") from exc
            if (current.st_dev, current.st_ino) != identity or not stat.S_ISSOCK(current.st_mode):
                raise PermissionError("storage socket endpoint changed")
            os.unlink(self._configuration.socket_path.name, dir_fd=directories.endpoint_parent)
        finally:
            stack.close()

    def __enter__(self) -> StorageRecordListener:
        """Start the endpoint and return its single owner."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close this listener without suppressing caller failures."""
        self.stop()
