# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Desktop backend without bundled interface resources

"""Refuse a real executable package copy whose built interface is absent."""

from __future__ import annotations

import fcntl
import hashlib
import inspect
import os
import selectors
import shutil
import socket
import subprocess
import sys
from pathlib import Path

from sc_neurocore.studio.desktop.runtime import DesktopBackend
from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from tests.test_studio_desktop_runtime import _environment


def test_package_without_built_interface_refuses_before_readiness(tmp_path: Path) -> None:
    """Run the unchanged backend from an actual package copy without frontend assets.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Private package location and user directories, without a checkout UI fallback.
    """
    env = _environment(tmp_path)
    source = Path(inspect.getfile(DesktopBackend)).resolve().parents[2]
    prefix = tmp_path / "package-only"
    package = prefix / "sc_neurocore"
    shutil.copytree(
        source,
        package,
        symlinks=True,
        ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", ".venv", ".pixi", ".git", "target", "frontend_dist"
        ),
    )
    backend = package / "studio/desktop/backend.py"
    digest = hashlib.sha256(backend.read_bytes()).hexdigest()
    assert digest == hashlib.sha256((source / "studio/desktop/backend.py").read_bytes()).hexdigest()
    retained = Path(env["HOME"]) / "research.bin"
    retained.write_bytes(b"retained experiment")
    owner = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            "import hashlib,inspect,sys; from pathlib import Path; "
            "prefix=Path(sys.argv[1]).resolve(); sys.path.insert(0,str(prefix)); "
            "from sc_neurocore.studio.desktop.runtime import DesktopBackend; "
            "source=Path(inspect.getfile(DesktopBackend)).resolve(); "
            "assert source.is_relative_to(prefix); "
            "assert hashlib.sha256(source.read_bytes()).hexdigest()==sys.argv[2]; "
            "DesktopBackend().start()",
            str(prefix),
            hashlib.sha256((package / "studio/desktop/runtime.py").read_bytes()).hexdigest(),
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert owner.returncode == 1, owner.stderr
    assert "DesktopStartRefused" in owner.stderr
    assert "This installation has no built Studio interface." in owner.stderr
    assert not Path(env["XDG_CONFIG_HOME"]).exists()
    assert not Path(env["XDG_STATE_HOME"]).exists()
    assert retained.read_bytes() == b"retained experiment"
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener,
        (tmp_path / "backend.lock").open("a+b") as lock,
    ):
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        address = listener.getsockname()
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reader, writer = os.pipe()
        command = [
            sys.executable,
            "-I",
            "-B",
            "-c",
            "import hashlib,sys; from pathlib import Path; "
            "prefix=Path(sys.argv.pop(1)).resolve(); sys.path.insert(0,str(prefix)); "
            "from sc_neurocore.studio.desktop import backend; "
            "source=Path(backend.__file__).resolve(); assert source.is_relative_to(prefix); "
            "assert hashlib.sha256(source.read_bytes()).hexdigest()==sys.argv.pop(1); "
            "raise SystemExit(backend.main())",
            str(prefix),
            digest,
            "--socket-fd",
            str(listener.fileno()),
            "--ready-fd",
            str(writer),
            "--lock-fd",
            str(lock.fileno()),
            "--parent",
            supervisor_identity(),
            "--nonce=missing-interface",
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
            assert selector.select(1), "Missing interface left the readiness pipe open."
            assert readiness.read(4097) == b""
        assert result.returncode == 1, result.stderr
        assert "DesktopStartRefused" in result.stderr, result.stderr
        assert "This installation has no built Studio interface." in result.stderr
        assert listener.getsockname() == address
        assert listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) == 1
        assert retained.read_bytes() == b"retained experiment"
