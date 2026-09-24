# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — storage listener lifecycle and real Unix boundary

"""Exercise the public listener with real paths, sockets and ledger custody."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import socket
import stat
import tempfile

import pytest

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.storage_listener import StorageRecordListener
from tests.studio_storage_listener_support import boundary as _boundary


def _held_directories(*paths: Path) -> set[int]:
    expected = {str(path) for path in paths}
    held: set[int] = set()
    for entry in Path("/proc/self/fd").iterdir():
        try:
            if os.readlink(entry) in expected:
                held.add(int(entry.name))
        except FileNotFoundError:
            continue
    return held


def test_start_stop_restart_owns_only_its_socket_and_descriptors(tmp_path: Path) -> None:
    """A real endpoint can restart without changing ledger bytes or leaking handles."""
    config, ledger, gateway = _boundary(tmp_path)
    before = ledger.path.read_bytes()
    listener = StorageRecordListener(config, ledger=ledger, gateway=gateway)
    baseline = _held_directories(config.authority_root, config.socket_path.parent)
    try:
        for _ in range(2):
            with listener:
                assert config.socket_path.is_socket()
                assert stat.S_IMODE(config.socket_path.stat().st_mode) == 0o660
                assert (
                    len(
                        _held_directories(config.authority_root, config.socket_path.parent)
                        - baseline
                    )
                    == 2
                )
                with pytest.raises(RuntimeError, match="already started"):
                    listener.start()
                with pytest.raises(TimeoutError):
                    listener.serve_once()
            assert not config.socket_path.exists()
            assert _held_directories(config.authority_root, config.socket_path.parent) == baseline
        listener.stop()
        assert ledger.path.read_bytes() == before
        assert not config.spool_root.exists()
    finally:
        listener.stop()
        ledger.close()


def test_existing_endpoint_is_never_adopted_or_unlinked() -> None:
    """Pre-existing regular files and stale sockets stay owned by their creator."""
    # A short base keeps the endpoint within the Unix socket path limit even
    # under a parallel test worker's longer temporary directory.
    base = Path(tempfile.mkdtemp(prefix="scs"))
    try:
        _occupied_endpoints_refuse(base)
    finally:
        shutil.rmtree(base)


def _occupied_endpoints_refuse(base: Path) -> None:
    for occupied in ("file", "stale-socket"):
        case = base / occupied
        case.mkdir()
        config, ledger, gateway = _boundary(case)
        if occupied == "file":
            config.socket_path.write_bytes(b"foreign")
        else:
            with socket.socket(socket.AF_UNIX) as foreign:
                foreign.bind(str(config.socket_path))
        before = config.socket_path.lstat()
        listener = StorageRecordListener(config, ledger=ledger, gateway=gateway)
        try:
            with pytest.raises(FileExistsError):
                listener.start()
            assert config.socket_path.lstat().st_ino == before.st_ino
            assert config.socket_path.lstat().st_mode == before.st_mode
            if occupied == "file":
                assert config.socket_path.read_bytes() == b"foreign"
        finally:
            listener.stop()
            ledger.close()


def test_untrusted_same_uid_peer_is_rejected_before_record_lookup(tmp_path: Path) -> None:
    """The listener remains live after kernel peer refusal on a real Unix socket."""
    config, ledger, gateway = _boundary(tmp_path, timeout=1.0)
    before = ledger.path.read_bytes()
    try:
        with StorageRecordListener(config, ledger=ledger, gateway=gateway) as listener:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(listener.serve_once)
                with socket.socket(socket.AF_UNIX) as client:
                    client.connect(str(config.socket_path))
                    client.settimeout(1)
                    try:
                        assert client.recv(1) == b""
                    except ConnectionResetError:
                        pass
                with pytest.raises(PermissionError, match="peer verification refused"):
                    future.result(timeout=2)
            assert config.socket_path.is_socket()
        assert ledger.path.read_bytes() == before
    finally:
        ledger.close()


def test_unsafe_endpoint_parent_refuses_without_creating_socket(tmp_path: Path) -> None:
    """The listener adds the directory restrictions needed for a distinct UID API."""
    for mode in (0o755, 0o700):
        case = tmp_path / f"mode-{mode:o}"
        case.mkdir()
        config, ledger, gateway = _boundary(case)
        config.socket_path.parent.chmod(mode)
        listener = StorageRecordListener(config, ledger=ledger, gateway=gateway)
        try:
            with pytest.raises(PermissionError, match="group traversal"):
                listener.start()
            assert stat.S_IMODE(config.socket_path.parent.stat().st_mode) == mode
            assert not config.socket_path.exists()
        finally:
            listener.stop()
            ledger.close()


def test_replaced_socket_is_not_unlinked_on_stop(tmp_path: Path) -> None:
    """A substituted file survives teardown while the original listener closes."""
    config, ledger, gateway = _boundary(tmp_path)
    listener = StorageRecordListener(config, ledger=ledger, gateway=gateway)
    retained = config.socket_path.with_name("retained.sock")
    try:
        listener.start()
        config.socket_path.rename(retained)
        config.socket_path.write_bytes(b"foreign")
        with pytest.raises(PermissionError, match="endpoint changed"):
            listener.stop()
        assert config.socket_path.read_bytes() == b"foreign"
        assert retained.is_socket()
        listener.stop()
    finally:
        listener.stop()
        retained.unlink(missing_ok=True)
        ledger.close()


def test_socket_mode_drift_refuses_client_and_restores_safe_teardown(tmp_path: Path) -> None:
    """A live endpoint with changed permissions cannot be served as trusted."""
    config, ledger, gateway = _boundary(tmp_path)
    listener = StorageRecordListener(config, ledger=ledger, gateway=gateway)
    before = ledger.path.read_bytes()
    try:
        listener.start()
        config.socket_path.chmod(0o666)
        with pytest.raises(PermissionError, match="endpoint changed"):
            listener.serve_once()
        assert ledger.path.read_bytes() == before
        config.socket_path.chmod(0o660)
        listener.stop()
        assert not config.socket_path.exists()
    finally:
        listener.stop()
        ledger.close()


def test_parent_write_permission_drift_refuses_accepted_work(tmp_path: Path) -> None:
    """A group-writable endpoint directory cannot remain a trusted channel."""
    config, ledger, gateway = _boundary(tmp_path)
    listener = StorageRecordListener(config, ledger=ledger, gateway=gateway)
    before = ledger.path.read_bytes()
    try:
        listener.start()
        config.socket_path.parent.chmod(0o2770)
        with pytest.raises(PermissionError, match="parent permissions changed"):
            listener.serve_once()
        assert ledger.path.read_bytes() == before
        config.socket_path.parent.chmod(0o2750)
        listener.stop()
        assert not config.socket_path.exists()
    finally:
        listener.stop()
        ledger.close()


def test_authority_permission_drift_refuses_accepted_work(tmp_path: Path) -> None:
    """A live ledger directory that becomes readable outside service is unsafe."""
    config, ledger, gateway = _boundary(tmp_path)
    listener = StorageRecordListener(config, ledger=ledger, gateway=gateway)
    before = ledger.path.read_bytes()
    try:
        listener.start()
        config.authority_root.chmod(0o750)
        with pytest.raises(PermissionError, match="authority changed"):
            listener.serve_once()
        assert ledger.path.read_bytes() == before
        config.authority_root.chmod(0o700)
        listener.stop()
        assert not config.socket_path.exists()
    finally:
        listener.stop()
        ledger.close()


def test_replaced_parent_refuses_before_accept_and_releases_owned_socket(tmp_path: Path) -> None:
    """Descriptor-pinned cleanup never touches a replacement directory."""
    config, ledger, gateway = _boundary(tmp_path)
    listener = StorageRecordListener(config, ledger=ledger, gateway=gateway)
    retained = tmp_path / "retained-endpoint"
    try:
        listener.start()
        config.socket_path.parent.rename(retained)
        config.socket_path.parent.mkdir(mode=0o2750)
        config.socket_path.parent.chmod(0o2750)
        with pytest.raises(PermissionError, match="endpoint is unavailable"):
            listener.serve_once()
        listener.stop()
        assert not (retained / config.socket_path.name).exists()
        assert list(config.socket_path.parent.iterdir()) == []
    finally:
        listener.stop()
        ledger.close()


def test_ledger_bound_to_other_authority_refuses_without_socket(tmp_path: Path) -> None:
    """A listener cannot be pointed at a second embedded database."""
    config, original, gateway = _boundary(tmp_path)
    other = StudioJobLedger(root=tmp_path / "other-authority")
    listener = StorageRecordListener(config, ledger=other, gateway=gateway)
    try:
        with pytest.raises(PermissionError, match="outside"):
            listener.start()
        assert not config.socket_path.exists()
        assert original.path.exists()
    finally:
        listener.stop()
        other.close()
        original.close()


def test_socket_group_other_than_its_parent_refuses_and_removes_the_socket(
    tmp_path: Path,
) -> None:
    """Without set-group-ID a socket takes the service's group; the API could not connect."""
    config, ledger, gateway = _boundary(tmp_path)
    parent = config.socket_path.parent
    other = next(group for group in os.getgroups() if group != os.getegid())
    os.chown(parent, -1, other)
    parent.chmod(0o750)
    listener = StorageRecordListener(config, ledger=ledger, gateway=gateway)
    try:
        with pytest.raises(PermissionError, match="group differs from its parent"):
            listener.start()
        assert not config.socket_path.exists()
    finally:
        listener.stop()
        ledger.close()
