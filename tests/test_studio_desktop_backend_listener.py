# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real desktop backend listener refusals

"""Refuse unsupported inherited sockets without a readiness record or owner loss."""

from __future__ import annotations

import fcntl
import json
import os
import secrets
import selectors
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from tests.test_studio_desktop_runtime import _environment


@pytest.mark.parametrize(
    ("family", "kind", "host", "function_mode"),
    [
        pytest.param(socket.AF_INET6, socket.SOCK_STREAM, "::1", True, id="ipv6-public-main"),
        pytest.param(socket.AF_UNIX, socket.SOCK_STREAM, None, False, id="unix-module-entry"),
        pytest.param(socket.AF_INET, socket.SOCK_DGRAM, "127.0.0.1", False, id="datagram"),
        pytest.param(socket.AF_INET, socket.SOCK_STREAM, "127.0.0.2", False, id="other-address"),
    ],
)
def test_invalid_listener_is_refused_before_readiness(
    tmp_path: Path,
    family: socket.AddressFamily,
    kind: socket.SocketKind,
    host: str | None,
    function_mode: bool,
) -> None:
    """Reject a real unsupported socket through the actual backend command.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Private user profile and retained experiment data.
    family : socket.AddressFamily
        Actual kernel socket family inherited by the backend.
    kind : socket.SocketKind
        Stream or datagram transport, never replaced by a test double.
    host : str or None
        Real IP bind address, or an ephemeral abstract Unix endpoint.
    function_mode : bool
        Invoke the public main function or the ordinary module entry point.
    """
    env = _environment(tmp_path)
    retained = Path(env["HOME"]) / "research.bin"
    retained.write_bytes(b"retained experiment")
    with socket.socket(family, kind) as listener, (tmp_path / "backend.lock").open("a+b") as lock:
        if family == socket.AF_UNIX:
            listener.bind("\0scn-studio-listener-" + secrets.token_hex(12))
        else:
            assert host is not None
            listener.bind((host, 0))
        if kind == socket.SOCK_STREAM:
            listener.listen()
        original_address = listener.getsockname()
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reader, writer = os.pipe()
        if function_mode:
            entry = [
                "-c",
                "from sc_neurocore.studio.desktop.backend import main; raise SystemExit(main())",
            ]
        else:
            entry = ["-m", "sc_neurocore.studio.desktop.backend"]
        command = [
            sys.executable,
            "-I",
            "-B",
            *entry,
            "--socket-fd",
            str(listener.fileno()),
            "--ready-fd",
            str(writer),
            "--lock-fd",
            str(lock.fileno()),
            "--parent",
            supervisor_identity(),
            "--nonce=listener-refusal",
        ]
        try:
            result = subprocess.run(
                command,
                env=env,
                cwd=tmp_path,
                pass_fds=(listener.fileno(), writer, lock.fileno()),
                capture_output=True,
                text=True,
                timeout=60,
                start_new_session=True,
            )
        finally:
            os.close(writer)
        with (
            os.fdopen(reader, "rb", buffering=0) as readiness,
            selectors.DefaultSelector() as selector,
        ):
            selector.register(readiness, selectors.EVENT_READ)
            assert selector.select(1), "Refused startup retained its readiness descriptor."
            assert readiness.read(4097) == b""
        (tmp_path / "refusal.json").write_text(
            json.dumps(
                {"exit": result.returncode, "stdout": result.stdout, "stderr": result.stderr},
                indent=2,
            )
        )
        assert result.returncode == 1, result.stderr
        assert "DesktopStartRefused" in result.stderr, result.stderr
        assert "requires its owned loopback listener" in result.stderr
        assert "ValueError" not in result.stderr
        assert listener.getsockname() == original_address
        assert listener.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) == kind
        assert retained.read_bytes() == b"retained experiment"
