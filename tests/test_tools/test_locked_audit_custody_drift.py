# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Lock audit input custody and unavailable inventory contracts

"""Refuse audits whose committed inputs change, are unknown or cannot bound a report."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

import pytest

from tools.security_scan.locked_dependency_audit import run_locked_audit

ROOT = Path(__file__).resolve().parents[2]


def _checkout(directory: Path, lock_name: str, lock_bytes: bytes) -> Path:
    """Create a real Git checkout with the maintained profiles and one tracked lock.

    Parameters
    ----------
    directory : Path
        Fresh directory that becomes the independent checkout.
    lock_name : str
        File name that selects the audited ecosystem.
    lock_bytes : bytes
        Exact tracked lock content.

    Returns
    -------
    Path
        The tracked lock inside the new checkout.
    """
    directory.mkdir()
    shutil.copytree(ROOT / "requirements", directory / "requirements")
    lock = directory / lock_name
    lock.write_bytes(lock_bytes)
    subprocess.run(
        ["git", "init", "--initial-branch=main", str(directory)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    subprocess.run(
        ["git", "add", "requirements", lock_name],
        cwd=directory,
        check=True,
        capture_output=True,
        timeout=30,
    )
    return lock


def test_public_api_refuses_an_unknown_ecosystem_before_reading_inputs(tmp_path: Path) -> None:
    """An ecosystem outside the maintained set fails closed with a retained summary.

    Parameters
    ----------
    tmp_path : Path
        Directory without Git custody; it must not be consulted at all.
    """
    target = tmp_path / "packet"
    report = run_locked_audit(tmp_path, target, "conda")
    assert report["passed"] is False and report["errors"] == ["ValueError"]
    assert report["inputs"] == [] and report["audits"] == []
    assert json.loads((target / "summary.json").read_text()) == report


def test_input_changed_during_a_real_request_blocks_the_audit(tmp_path: Path) -> None:
    """A tracked lock that changes while its advisory request is open fails the audit.

    Parameters
    ----------
    tmp_path : Path
        Fresh independent Git checkout and retained CLI audit packet.

    Notes
    -----
    The audit's real request goes to a local TCP peer named as its proxy. The
    peer accepts the connection, reads the tunnel request, sends nothing and
    closes. No HTTP or advisory response is substituted. The accepted
    connection proves that the audit is inside its request, so the lock is
    rewritten before the audit can compare its inputs again.
    """
    source = ROOT / "src/sc_neurocore/accel/julia/sc_compte_wm_network/Manifest.toml"
    original = source.read_bytes()
    lock = _checkout(tmp_path / "checkout", "Manifest.toml", original)
    target = tmp_path / "packet"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(30)
        proxy = f"http://127.0.0.1:{listener.getsockname()[1]}"
        process = subprocess.Popen(
            [
                sys.executable,
                "-B",
                "-m",
                "tools.security_scan.locked_dependency_audit",
                "--repo-root",
                str(lock.parent),
                "--ecosystem",
                "julia",
                "--output-dir",
                str(target),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
                "HTTPS_PROXY": proxy,
                "https_proxy": proxy,
                "HTTP_PROXY": proxy,
                "http_proxy": proxy,
                "NO_PROXY": "",
                "no_proxy": "",
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            peer, _ = listener.accept()
            with peer:
                peer.settimeout(30)
                received = b""
                while b"\r\n\r\n" not in received:
                    chunk = peer.recv(4096)
                    assert chunk, "the audit closed its request before sending it"
                    received += chunk
                lock.write_bytes(original + b"\n")
            process.communicate(timeout=45)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
    assert received.startswith(b"CONNECT api.osv.dev:443 ")
    assert process.returncode == 1
    report = json.loads((target / "summary.json").read_text())
    assert report["passed"] is False and report["errors"] == ["ValueError"]
    assert report["audits"] == [
        {"path": "Manifest.toml", "passed": False, "error_type": "URLError"}
    ]
    bound = {row["path"]: row["sha256"] for row in report["inputs"]}
    assert bound["Manifest.toml"] == hashlib.sha256(original).hexdigest()
    assert hashlib.sha256(lock.read_bytes()).hexdigest() != bound["Manifest.toml"]
    receipt = json.loads((target / "lock-1/response-1-receipt.json").read_text())
    assert receipt["http_status"] is None
    assert receipt["error_type"] == "URLError"
    assert receipt["reason_type"] == "RemoteDisconnected"
    assert not (target / "lock-1/response-1.json").exists()


@pytest.mark.parametrize(
    ("lock_bytes", "error_type"),
    [
        (b"version: 6\n", "ValueError"),
        (b"packages: {}\n", "ValueError"),
        (b"packages: [\n", "ParserError"),
    ],
)
def test_public_cli_refuses_a_pixi_lock_without_a_package_inventory(
    tmp_path: Path, lock_bytes: bytes, error_type: str
) -> None:
    """Refuse a tracked Pixi lock that cannot bound report coverage before any scan.

    Parameters
    ----------
    tmp_path : Path
        Fresh real Git checkout and retained refusal packet.
    lock_bytes : bytes
        Tracked lock without a readable package record list.
    error_type : str
        Refusal type retained for the lock.

    Notes
    -----
    The refusal precedes scanner execution, so no scanner is started and no
    scanner response is substituted. The tracked lock bytes stay unchanged.
    """
    lock = _checkout(tmp_path / "checkout", "pixi.lock", lock_bytes)
    target = tmp_path / "packet"
    process = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "tools.security_scan.locked_dependency_audit",
            "--repo-root",
            str(lock.parent),
            "--ecosystem",
            "pixi",
            "--output-dir",
            str(target),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert process.returncode == 1
    report = json.loads((target / "summary.json").read_text())
    assert report["passed"] is False and report["errors"] == []
    assert report["audits"] == [{"path": "pixi.lock", "passed": False, "error_type": error_type}]
    assert not (target / "lock-1").exists()
    assert lock.read_bytes() == lock_bytes
