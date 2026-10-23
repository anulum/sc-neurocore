# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Desktop backend admission and process custody

"""Reserve a stable origin, admit the actual backend, and reap only owned processes."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import secrets
import selectors
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from sc_neurocore import __version__
from sc_neurocore.studio.desktop import DesktopStartRefused
from sc_neurocore.studio.desktop.configuration import (
    DesktopConfiguration,
    read_configuration,
    write_configuration,
)
from sc_neurocore.studio.desktop.paths import DesktopPaths
from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.jobs_reaper import ReapReport, reap_process_group
from sc_neurocore.studio.project_paths import PROJECT_ROOT_ENV


@dataclass(frozen=True, slots=True)
class DesktopInstance:
    """Admitted backend identity for the window that owns it.

    Attributes
    ----------
    pid : int
        Direct owned child process, also its session/group leader.
    origin : str
        Stable IPv4 loopback origin used by HTTP, WebSocket and browser storage.
    version, ui_sha256 : str
        Source package version and exact served interface entry digest.
    project_root : pathlib.Path
        Selected project namespace, outside installation resources.
    """

    pid: int
    origin: str
    version: str
    ui_sha256: str
    project_root: Path


class DesktopBackend:
    """Own one backend and retain custody across startup failure and window exit."""

    def __init__(self, environ: Mapping[str, str] | None = None) -> None:
        """Resolve one profile without starting a process or creating directories.

        Parameters
        ----------
        environ : mapping of str to str, optional
            Actual user environment by default; copied without mutating the caller.
        """
        self.environ = dict(os.environ if environ is None else environ)
        self.paths = DesktopPaths.from_environment(self.environ)
        self._process: subprocess.Popen[bytes] | None = None
        self._report: ReapReport | None = None
        self._cancelled = threading.Event()

    @property
    def returncode(self) -> int | None:
        """Return child exit status, or ``None`` before startup or while running."""
        process = self._process
        if process is not None:
            return process.poll()
        return None if self._report is None else self._report.returncode

    def cancel_startup(self) -> None:
        """Request startup cancellation; the starting thread retains process cleanup."""
        self._cancelled.set()

    def start(self, *, timeout_seconds: float = 30.0) -> DesktopInstance:
        """Start and admit the actual backend before retaining its browser origin.

        Parameters
        ----------
        timeout_seconds : float
            Finite positive readiness deadline, including actual HTTP probes.

        Returns
        -------
        DesktopInstance
            Matched child/package/UI identity after real HTTP admission.

        Raises
        ------
        DesktopStartRefused
            The profile is busy, an existing port is occupied, the interface is
            absent, or startup/readiness/configuration persistence fails.
        ValueError
            The deadline is not finite and positive or this owner already started.
        """
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Desktop startup timeout must be finite and positive.")
        if self._process is not None or self._report is not None:
            raise ValueError("A desktop backend owner can only be started once.")
        deadline = time.monotonic() + timeout_seconds
        from sc_neurocore.studio.frontend_resources import (
            studio_frontend_candidates,
            studio_frontend_dir,
        )

        app_file = Path(__file__).resolve().parents[1] / "app.py"
        interface = studio_frontend_dir(studio_frontend_candidates(str(app_file)))
        if interface is None:
            raise DesktopStartRefused("This installation has no built Studio interface.")
        digest = hashlib.sha256((interface / "index.html").read_bytes()).hexdigest()
        previous = read_configuration(self.paths.config / "desktop.json")
        self.paths.project_root(self.environ, None if previous is None else previous.project_root)
        self.paths.prepare()
        try:
            with (self.paths.state / "backend.lock").open("a+b") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise DesktopStartRefused(
                        "Studio is already running for this desktop profile."
                    ) from None
                path = self.paths.config / "desktop.json"
                retained = read_configuration(path)
                project = self.paths.project_root(
                    self.environ, None if retained is None else retained.project_root
                )
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    try:
                        listener.bind(("127.0.0.1", 0 if retained is None else retained.port))
                    except OSError:
                        raise DesktopStartRefused(
                            "The saved Studio port is in use or unavailable."
                        ) from None
                    listener.listen(128)
                    port = int(listener.getsockname()[1])
                    origin = f"http://127.0.0.1:{port}"
                    env = self._environment(origin, project)
                    nonce = secrets.token_urlsafe(32)
                    if self._cancelled.is_set():
                        raise DesktopStartRefused("Studio startup was cancelled.")
                    read_fd, write_fd = os.pipe()
                    with (
                        os.fdopen(read_fd, "rb", buffering=0) as reader,
                        os.fdopen(write_fd, "wb", buffering=0) as writer,
                        (self.paths.state / "backend.log").open("ab") as log,
                    ):
                        self._process = subprocess.Popen(
                            [
                                sys.executable,
                                "-I",
                                "-B",
                                "-m",
                                "sc_neurocore.studio.desktop.backend",
                                "--socket-fd",
                                str(listener.fileno()),
                                "--ready-fd",
                                str(writer.fileno()),
                                "--lock-fd",
                                str(lock.fileno()),
                                "--parent",
                                supervisor_identity(),
                                "--nonce=" + nonce,
                            ],
                            env=env,
                            pass_fds=(listener.fileno(), writer.fileno(), lock.fileno()),
                            start_new_session=True,
                            stdin=subprocess.DEVNULL,
                            stdout=log,
                            stderr=log,
                        )
                        writer.close()
                        self._admit(reader, nonce, digest, deadline)
                        _probe(origin, digest, deadline)
                        if self._cancelled.is_set():
                            raise DesktopStartRefused("Studio startup was cancelled.")
                        try:
                            write_configuration(path, DesktopConfiguration(port, project))
                        except OSError:
                            raise DesktopStartRefused(
                                "Studio could not save its desktop configuration."
                            ) from None
                        return DesktopInstance(
                            self._process.pid, origin, __version__, digest, project
                        )
        except BaseException:
            self.close()
            raise

    def close(self) -> ReapReport | None:
        """Stop the owned group and preserve an honest report of any survivors.

        Returns
        -------
        ReapReport or None
            Verified group cleanup, an unreaped report, or no child was started.
            Repeated calls after successful cleanup never signal a recycled PID.
        """
        if self._process is None:
            return self._report
        self._report = reap_process_group(
            self._process,
            owned_group_id=self._process.pid,
            terminate_grace_seconds=12.0,
            kill_grace_seconds=5.0,
        )
        if self._report.reaped:
            self._process = None
        return self._report

    def _environment(self, origin: str, project: Path) -> dict[str, str]:
        """Bind this local profile to its reserved origin and persistent storage."""
        if (
            self.environ.get("SC_NEUROCORE_STUDIO_DEPLOYMENT_PROFILE", "development")
            != "development"
        ):
            raise DesktopStartRefused(
                "The desktop launcher requires the local development profile."
            )
        if self.environ.get("SC_NEUROCORE_STUDIO_STORAGE_MODE", "embedded") != "embedded":
            raise DesktopStartRefused("The desktop launcher requires embedded local storage.")
        env = dict(self.environ)
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        env.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                PROJECT_ROOT_ENV: str(project),
                "SC_NEUROCORE_STUDIO_CORS_ORIGINS": origin,
                "SC_NEUROCORE_STUDIO_WEBSOCKET_ALLOWED_ORIGINS": origin,
                "SC_NEUROCORE_STUDIO_ALLOWED_HOSTS": "127.0.0.1",
                "SC_NEUROCORE_STUDIO_JOB_ROOT": str(self.paths.state / "jobs"),
                "SC_NEUROCORE_STUDIO_AUDIT_LOG_PATH": str(self.paths.state / "audit.jsonl"),
                "SC_NEUROCORE_STUDIO_BENCHMARK_DATABANK": str(self.paths.data / "benchmarks.jsonl"),
            }
        )
        return env

    def _admit(self, stream: BinaryIO, nonce: str, digest: str, deadline: float) -> None:
        """Read one bounded private readiness record from the actual child."""
        payload = b""
        with selectors.DefaultSelector() as selector:
            selector.register(stream, selectors.EVENT_READ)
            while b"\n" not in payload:
                if self._cancelled.is_set():
                    raise DesktopStartRefused("Studio startup was cancelled.")
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self.returncode is not None:
                    raise DesktopStartRefused("The Studio backend did not become ready.")
                if not selector.select(min(remaining, 0.1)):
                    continue
                chunk = stream.read(4097 - len(payload))
                if not chunk or len(payload) + len(chunk) > 4096:
                    raise DesktopStartRefused(
                        "The Studio backend readiness record is missing or invalid."
                    )
                payload += chunk
        try:
            value: object = json.loads(payload)
        except (ValueError, UnicodeError):
            raise DesktopStartRefused(
                "The Studio backend readiness record is not valid JSON."
            ) from None
        if (
            not isinstance(value, dict)
            or self._process is None
            or value
            != {
                "schema_version": "studio.desktop.ready.v1",
                "pid": self._process.pid,
                "nonce": nonce,
                "version": __version__,
                "ui_sha256": digest,
            }
        ):
            raise DesktopStartRefused(
                "The Studio backend does not match this desktop installation."
            )


def _probe(origin: str, digest: str, deadline: float) -> None:
    """Match real health and served UI bytes without following environment proxies."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise DesktopStartRefused("The Studio backend readiness deadline expired.")
        with opener.open(origin + "/api/health", timeout=min(remaining, 2.0)) as response:
            health = json.loads(response.read(4097))
        if health != {"status": "ok"}:
            raise DesktopStartRefused("The Studio backend did not report healthy startup.")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise DesktopStartRefused("The Studio backend readiness deadline expired.")
        with opener.open(
            origin + "/studios/sc-neurocore/", timeout=min(remaining, 2.0)
        ) as response:
            page = response.read(1_048_577)
        if hashlib.sha256(page).hexdigest() != digest:
            raise DesktopStartRefused(
                "The served Studio interface does not match this installation."
            )
    except DesktopStartRefused:
        raise
    except (OSError, ValueError):
        raise DesktopStartRefused("The Studio backend HTTP admission failed.") from None
