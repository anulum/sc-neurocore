# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Desktop backend public HTTP and process lifecycle

"""Exercise descriptor-owned real servers, retained origins and project recovery."""

from __future__ import annotations

import json
import hashlib
import fcntl
import inspect
import os
import secrets
import socket
import subprocess
import sys
import textwrap
import urllib.request
from pathlib import Path

import pytest

from sc_neurocore.studio.desktop import DesktopStartRefused
from sc_neurocore.studio.desktop.runtime import DesktopBackend


def _environment(root: Path) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SC_NEUROCORE_STUDIO_")
    }
    home = root / "home"
    temporary = root / "tmp"
    home.mkdir()
    temporary.mkdir()
    env.update(
        HOME=str(home),
        TMPDIR=str(temporary),
        PYTHONDONTWRITEBYTECODE="1",
        XDG_DATA_HOME=str(root / "data"),
        XDG_CONFIG_HOME=str(root / "config"),
        XDG_STATE_HOME=str(root / "state"),
        XDG_CACHE_HOME=str(root / "cache"),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    return env


def _request(origin: str, path: str, body: dict[str, object] | None = None) -> object:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        origin + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Origin": origin, "Content-Type": "application/json"},
    )
    with opener.open(request, timeout=5) as response:
        result: object = json.loads(response.read(1_048_577))
    return result


def test_backend_keeps_scientific_application_in_its_owned_child(tmp_path: Path) -> None:
    """Serve an actual seeded model without importing its application in the owner.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Private profile inherited by a fresh isolated launcher interpreter.
    """
    command = textwrap.dedent(
        """
        import json
        import hashlib
        import inspect
        import sys
        import urllib.request
        from pathlib import Path
        from sc_neurocore.studio.desktop.runtime import DesktopBackend

        source = Path(inspect.getfile(DesktopBackend))
        assert str(source.resolve()) == sys.argv[1], source
        assert hashlib.sha256(source.read_bytes()).hexdigest() == sys.argv[2]
        assert 'sc_neurocore.studio.app' not in sys.modules
        backend = DesktopBackend()
        try:
            instance = backend.start()
            request = urllib.request.Request(
                instance.origin + '/api/models/simulate',
                data=json.dumps({
                    'model_name': 'SCStochasticRateAdaptationNeuron',
                    'params': {'tau_a': 37}, 'dt': 0.5, 'duration': 20,
                    'current': 10, 'seed': 9223372036854775807,
                    'trial': 'replay',
                }).encode(),
                headers={'Origin': instance.origin, 'Content-Type': 'application/json'},
            )
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=5) as response:
                result = json.loads(response.read(1048577))
            assert result['model_name'] == 'SCStochasticRateAdaptationNeuron', result
            assert result['n_steps'] == 40, result
        finally:
            report = backend.close()
            assert report is not None and report.reaped, report
            assert report.returncode == 0, (backend.paths.state / 'backend.log').read_text()
        assert 'sc_neurocore.studio.app' not in sys.modules, (
            'The desktop owner imported the scientific application.'
        )
        assert 'sc_neurocore.studio.api' not in sys.modules
        """
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            command,
            str(Path(inspect.getfile(DesktopBackend)).resolve()),
            hashlib.sha256(Path(inspect.getfile(DesktopBackend)).read_bytes()).hexdigest(),
        ],
        env=_environment(tmp_path),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_backend_retains_actual_projects_and_origin_across_restart(tmp_path: Path) -> None:
    """Reopen exact project state through a restarted real backend at the same origin.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Disposable desktop configuration, user data and runtime state.
    """
    env = _environment(tmp_path)
    backend = DesktopBackend(env)
    assert backend.returncode is None and backend.close() is None
    try:
        first = backend.start()
        assert _request(first.origin, "/api/health") == {"status": "ok"}
        state = {"seed": "9223372036854775807", "equations": ["dv/dt = -v / 10"]}
        _request(first.origin, "/api/project/save", {"name": "desktop", "state": state})
        project = _request(first.origin, "/api/project/load/desktop")
        assert isinstance(project, dict)
        assert project["state"] == state
        assert first.project_root == backend.paths.data / "projects"
        assert not backend.paths.legacy_projects.exists()
    finally:
        report = backend.close()
        assert report is not None and report.reaped, report
        assert report.returncode == 0, (backend.paths.state / "backend.log").read_text()
    assert backend.returncode == 0
    with pytest.raises(ValueError, match="only be started once"):
        backend.start()
    resumed = DesktopBackend(env)
    try:
        second = resumed.start()
        assert second.origin == first.origin
        assert second.pid != first.pid
        assert second.ui_sha256 == first.ui_sha256
        assert second.project_root == first.project_root
        document = _request(second.origin, "/api/project/load/desktop")
        assert isinstance(document, dict)
        assert document["state"] == state
    finally:
        report = resumed.close()
        assert report is not None and report.reaped, report
        assert report.returncode == 0, (resumed.paths.state / "backend.log").read_text()
    assert resumed.close() is report


def test_busy_profile_and_occupied_saved_port_preserve_other_server(tmp_path: Path) -> None:
    """Refuse a second owner and occupied retained port without signalling either.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Disposable desktop profile with one admitted listening origin.
    """
    env = _environment(tmp_path)
    backend = DesktopBackend(env)
    try:
        instance = backend.start()
        before = (backend.paths.config / "desktop.json").read_bytes()
        with pytest.raises(DesktopStartRefused, match="already running"):
            DesktopBackend(env).start()
        assert _request(instance.origin, "/api/health") == {"status": "ok"}
        assert backend.returncode is None
    finally:
        report = backend.close()
        assert report is not None and report.reaped, report
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        occupied.bind(("127.0.0.1", int(instance.origin.rsplit(":", 1)[1])))
        occupied.listen()
        with pytest.raises(DesktopStartRefused, match="port is in use"):
            DesktopBackend(env).start()
        assert occupied.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) == 1
    assert (backend.paths.config / "desktop.json").read_bytes() == before


def test_leading_dash_readiness_nonce_reaches_actual_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Admit a real backend when the entropy source produces a leading dash.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Disposable user profile and actual backend state.
    monkeypatch : pytest.MonkeyPatch
        Constrain only the parent entropy input; the actual interpreter,
        descriptor handshake and HTTP application still execute unchanged.
    """
    monkeypatch.setattr(secrets, "token_urlsafe", lambda _length: "-" + "x" * 42)
    backend = DesktopBackend(_environment(tmp_path))
    try:
        instance = backend.start()
        assert _request(instance.origin, "/api/health") == {"status": "ok"}
    finally:
        report = backend.close()
        assert report is not None and report.reaped, report
        assert report.returncode == 0, (backend.paths.state / "backend.log").read_text()


@pytest.mark.parametrize(
    "prefix,message",
    [
        (b"{\n", "readiness record is not valid JSON"),
        (b"[]\n", "does not match this desktop installation"),
        (b"{}\n", "does not match this desktop installation"),
        (b" " * 4096, "readiness record is missing or invalid"),
    ],
    ids=["invalid-json", "non-object", "foreign-record", "oversized-record"],
)
def test_corrupted_readiness_collects_the_real_backend_and_allows_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefix: bytes, message: str
) -> None:
    """Refuse corrupted descriptor input while retaining actual process custody.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Disposable persistent profile with an existing user document.
    monkeypatch : pytest.MonkeyPatch
        One-call OS pipe input hook, restored before the real child is spawned.
    prefix : bytes
        Malformed, mismatched or size-exhausting bytes written into the real pipe.
    message : str
        Expected public admission refusal for these actual descriptor bytes.
    """
    env = _environment(tmp_path)
    retained = tmp_path / "home" / "retained.txt"
    retained.write_bytes(b"preserve work through readiness refusal\n")
    normal_pipe = os.pipe
    injected: list[tuple[int, int]] = []

    def pipe_with_prefix() -> tuple[int, int]:
        """Create the normal kernel pipe and prefix its actual input once."""
        reader, writer = normal_pipe()
        monkeypatch.setattr(os, "pipe", normal_pipe)
        injected.append((reader, writer))
        assert os.write(writer, prefix) == len(prefix)
        return reader, writer

    backend = DesktopBackend(env)
    monkeypatch.setattr(os, "pipe", pipe_with_prefix)
    with pytest.raises(DesktopStartRefused, match=message):
        backend.start()
    assert len(injected) == 1 and os.pipe is normal_pipe
    report = backend.close()
    assert report is not None and report.reaped and report.returncode is not None
    assert backend.returncode == report.returncode
    assert not (backend.paths.config / "desktop.json").exists()
    assert retained.read_bytes() == b"preserve work through readiness refusal\n"
    with (backend.paths.state / "backend.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    resumed = DesktopBackend(env)
    try:
        instance = resumed.start()
        assert _request(instance.origin, "/api/health") == {"status": "ok"}
        assert retained.read_bytes() == b"preserve work through readiness refusal\n"
    finally:
        recovered = resumed.close()
        assert recovered is not None and recovered.reaped and recovered.returncode == 0
