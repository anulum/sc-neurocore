# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — storage authority and lossy relays for generation supervision tests

"""A real storage authority on real sockets, and relays that really lose replies.

:class:`Authority` answers every connection the API opens through the
service's own dispatch (:func:`storage_dispatch.serve_operation`) over a real
SQLite ledger, exactly as the listener does after accepting a connection. :class:`LauncherRelay` is a real
Unix endpoint in front of the real launcher process. Either can drop a reply
after the real peer acted, or a request before it reached the peer: a lost
message on a real connection, not a replaced handler.
"""

from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
import socket
import threading
import time

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_shared_admission import SharedJobAdmission
from sc_neurocore.studio.platform.policy_models import InMemoryAuditSink
from sc_neurocore.studio.platform.policy_gateway import PolicyGateway
from sc_neurocore.studio.platform.storage_dispatch import StorageServices, serve_operation
from sc_neurocore.studio.platform.storage_named_admit import named_process_admission
from sc_neurocore.studio.platform.storage_finish_protocol import (
    decode_finish_request,
    decode_finish_response,
)
from sc_neurocore.studio.platform.storage_launcher_protocol import (
    LAUNCHER_MESSAGE_MAX_BYTES,
    LauncherResponse,
    decode_launcher_request,
    decode_launcher_response,
)
from sc_neurocore.studio.platform.storage_operation import classify_storage_operation
from sc_neurocore.studio.platform.storage_peer import read_verified_frame
from sc_neurocore.studio.platform.storage_transport import read_frame, write_frame

FRAME = 1 << 16
_WIRE = 10.0


def _deadline() -> float:
    return time.monotonic() + _WIRE


class Authority:
    """Serve each API connection with the real handlers in its own thread.

    ``lose`` maps ``start``, ``heartbeat`` or ``finish`` to how many of the
    next final replies of that operation are dropped after the handler acted.
    ``seen`` records every operation received, in order.
    """

    def __init__(self, ledger: StudioJobLedger, *, frame_max_bytes: int = FRAME) -> None:
        self.ledger = ledger
        self.audit = InMemoryAuditSink()
        self._directory = os.open(ledger.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        admission = SharedJobAdmission(ledger, max_concurrent=2, max_queued=0)
        self.services = StorageServices(
            ledger=ledger,
            gateway=PolicyGateway(self.audit),
            workspace="default",
            api_uid=os.getuid(),
            frame_max_bytes=frame_max_bytes,
            max_metadata_bytes=FRAME,
            max_seed_bytes=FRAME,
            max_seed_entries=16,
            max_manifest_bytes=FRAME,
            max_artifact_bytes=1 << 20,
            max_artifact_entries=64,
            admission=admission,
            admit_named=named_process_admission(admission, workspace="default"),
            authority_dirfd=self._directory,
        )
        self.lose: dict[str, int] = {}
        self.seen: list[str] = []
        #: Runs on each serving thread before its handler, e.g. to schedule a
        #: competing writer through that thread's SQLite trace hook.
        self.before: Callable[[StudioJobLedger], None] | None = None
        self.failures: list[BaseException] = []
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()

    def connect(self) -> socket.socket:
        """Return the API end of a new connection served by a real handler."""
        client, service = socket.socketpair()
        thread = threading.Thread(target=self._serve, args=(service,), daemon=True)
        self._threads.append(thread)
        thread.start()
        return client

    def _losing(self, name: str) -> bool:
        with self._lock:
            self.seen.append(name)
            if self.lose.get(name, 0) <= 0:
                return False
            self.lose[name] -= 1
            return True

    def _handle(self, channel: socket.socket, frame: bytes) -> None:
        if self.before is not None:
            self.before(self.ledger)
        serve_operation(channel, frame, self.services, deadline=_deadline())

    def _serve(self, channel: socket.socket) -> None:
        try:
            with channel:
                frame = read_verified_frame(
                    channel, expected_uid=os.getuid(), max_bytes=FRAME, deadline=_deadline()
                )
                operation = classify_storage_operation(frame)
                name = operation if operation == "finish" else json.loads(frame)["operation"]
                if not self._losing(name):
                    self._handle(channel, frame)
                    return
                self._relay_losing_reply(channel, frame, operation)
        except BaseException as exc:
            self.failures.append(exc)
        finally:
            self.ledger.close()

    def _relay_losing_reply(self, client: socket.socket, frame: bytes, operation: str) -> None:
        """Let the real handler act through a relay that drops its final reply."""
        relay, service = socket.socketpair()
        handler = threading.Thread(target=self._serve_inner, args=(service, frame), daemon=True)
        handler.start()
        with relay:
            reply = read_frame(relay, max_bytes=FRAME, deadline=_deadline())
            if operation == "finish":
                request = decode_finish_request(frame, max_bytes=FRAME)
                decoded = decode_finish_response(reply, request=request, max_bytes=FRAME)
                if decoded.reply == "ready":
                    write_frame(client, reply, max_bytes=FRAME, deadline=_deadline())
                    for artifact in request.artifacts:
                        if artifact.size_bytes:
                            payload = read_frame(client, max_bytes=FRAME, deadline=_deadline())
                            write_frame(relay, payload, max_bytes=FRAME, deadline=_deadline())
                    read_frame(relay, max_bytes=FRAME, deadline=_deadline())
        handler.join(timeout=_WIRE)

    def _serve_inner(self, channel: socket.socket, frame: bytes) -> None:
        try:
            self._handle(channel, frame)
        except BaseException as exc:
            self.failures.append(exc)
        finally:
            self.ledger.close()

    def join(self, *, expected: tuple[type[BaseException], ...] = ()) -> None:
        """Wait for every served connection; only ``expected`` failures may occur."""
        for thread in self._threads:
            thread.join(timeout=_WIRE)
            assert not thread.is_alive()
        assert [type(failure) for failure in self.failures] == list(expected)
        self.failures.clear()
        os.close(self._directory)


class LauncherRelay:
    """A real Unix endpoint in front of the launcher that can lose messages.

    ``lose_reply`` drops the next replies of an operation after the launcher
    acted; ``lose_request`` drops requests before they reach it.
    ``before_loss`` runs with the dropped reply before the connection closes.
    """

    def __init__(self, path: Path, target: Path) -> None:
        self.path = path
        self.target = target
        self.lose_reply: dict[str, int] = {}
        self.lose_request: dict[str, int] = {}
        self.before_loss: Callable[[LauncherResponse], None] | None = None
        self.seen: list[str] = []
        self.failures: list[BaseException] = []
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(str(path))
        self._listener.listen(8)
        self._listener.settimeout(0.2)
        self._stopping = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _take(self, table: dict[str, int], operation: str) -> bool:
        if table.get(operation, 0) <= 0:
            return False
        table[operation] -= 1
        return True

    def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                client, _ = self._listener.accept()
            except TimeoutError:
                continue
            try:
                with client:
                    self._forward(client)
            except BaseException as exc:
                self.failures.append(exc)

    def _forward(self, client: socket.socket) -> None:
        raw = read_frame(client, max_bytes=LAUNCHER_MESSAGE_MAX_BYTES, deadline=_deadline())
        request = decode_launcher_request(raw)
        self.seen.append(request.operation)
        if self._take(self.lose_request, request.operation):
            return
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as upstream:
            upstream.settimeout(_WIRE)
            upstream.connect(str(self.target))
            write_frame(upstream, raw, max_bytes=LAUNCHER_MESSAGE_MAX_BYTES, deadline=_deadline())
            reply = read_frame(upstream, max_bytes=LAUNCHER_MESSAGE_MAX_BYTES, deadline=_deadline())
        if self._take(self.lose_reply, request.operation):
            if self.before_loss is not None:
                self.before_loss(decode_launcher_response(reply, request=request))
            return
        write_frame(client, reply, max_bytes=LAUNCHER_MESSAGE_MAX_BYTES, deadline=_deadline())

    def close(self) -> None:
        """Stop relaying and require that no relayed exchange failed."""
        self._stopping.set()
        self._thread.join(timeout=_WIRE)
        self._listener.close()
        self.path.unlink(missing_ok=True)
        assert self.failures == []
