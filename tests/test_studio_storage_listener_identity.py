# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — storage listener ledger identity

"""Refuse replacement of the ledger's real authority after SQLite opens."""

import errno
from pathlib import Path
import os
import shutil
from threading import TIMEOUT_MAX

import pytest

from sc_neurocore.studio.platform.storage_listener import StorageRecordListener
from sc_neurocore.studio.platform.storage_configuration import StorageBoundaryConfiguration
from tests.studio_seccomp_support import SECCOMP_AVAILABLE, run_child
from tests.studio_storage_listener_support import boundary


def test_replaced_ledger_root_before_start_refuses_copied_database(tmp_path: Path) -> None:
    """A copied path cannot make an open connection to an old root authoritative."""
    config, ledger, gateway = boundary(tmp_path)
    retained = tmp_path / "original-authority"
    config.authority_root.rename(retained)
    config.authority_root.mkdir(mode=0o700)
    shutil.copyfile(retained / ledger.path.name, ledger.path)
    listener = StorageRecordListener(config, ledger=ledger, gateway=gateway)
    try:
        with pytest.raises(PermissionError, match="ledger identity changed"):
            listener.start()
        assert not config.socket_path.exists()
        assert (retained / ledger.path.name).is_file()
    finally:
        listener.stop()
        ledger.close()


def test_replaced_ledger_file_after_start_refuses_serving(tmp_path: Path) -> None:
    """A live endpoint cannot read through a substituted database pathname."""
    config, ledger, gateway = boundary(tmp_path)
    listener = StorageRecordListener(config, ledger=ledger, gateway=gateway)
    retained = config.authority_root / "retained-ledger.sqlite3"
    try:
        listener.start()
        ledger.path.rename(retained)
        shutil.copyfile(retained, ledger.path)
        with pytest.raises(PermissionError, match="ledger identity changed"):
            listener.serve_once()
        listener.stop()
        assert not config.socket_path.exists()
        assert retained.is_file()
    finally:
        listener.stop()
        ledger.close()


def test_missing_ledger_file_refuses_before_socket_bind(tmp_path: Path) -> None:
    """A vanished database cannot be silently recreated by listener startup."""
    config, ledger, gateway = boundary(tmp_path)
    ledger.path.unlink()
    listener = StorageRecordListener(config, ledger=ledger, gateway=gateway)
    try:
        with pytest.raises(PermissionError, match="ledger is unavailable"):
            listener.start()
        assert not config.socket_path.exists()
        assert not ledger.path.exists()
    finally:
        listener.stop()
        ledger.close()


@pytest.mark.skipif(not SECCOMP_AVAILABLE, reason="held system calls need Linux x86_64")
def test_socket_mode_failure_unlinks_only_own_endpoint_and_can_restart(tmp_path: Path) -> None:
    """A post-bind permission failure leaves no stale socket or held directory.

    The kernel refuses the socket's mode change once; the same listener then
    starts cleanly.
    """
    result = run_child(
        "import errno, sys\n"
        "from pathlib import Path\n"
        "from sc_neurocore.studio.platform.storage_listener import StorageRecordListener\n"
        "from tests.studio_storage_listener_support import boundary\n"
        "from tests.studio_syscall_support import finish, hold_system_calls\n"
        "config, ledger, gateway = boundary(Path(sys.argv[1]))\n"
        "listener = StorageRecordListener(config, ledger=ledger, gateway=gateway)\n"
        "refusing = [True]\n"
        "def decide(call):\n"
        "    if refusing[0] and call.text(1) == config.socket_path.name:\n"
        "        refusing[0] = False\n"
        "        return errno.EPERM\n"
        "    return None\n"
        "hold_system_calls(['fchmodat', 'fchmodat2'], decide)\n"
        "out = {}\n"
        "try:\n"
        "    listener.start()\n"
        "except PermissionError as refused:\n"
        "    out['refused'] = refused.errno\n"
        "out['left'] = config.socket_path.exists()\n"
        "with listener:\n"
        "    out['restarted'] = config.socket_path.is_socket()\n"
        "out['removed'] = not config.socket_path.exists()\n"
        "listener.stop()\n"
        "ledger.close()\n"
        "finish(out)\n",
        arguments=(str(tmp_path),),
    )
    assert result == {"refused": errno.EPERM, "left": False, "restarted": True, "removed": True}


def test_removed_socket_refuses_stop_and_releases_listener(tmp_path: Path) -> None:
    """Teardown reports a missing endpoint while closing its owned descriptors."""
    config, ledger, gateway = boundary(tmp_path)
    listener = StorageRecordListener(config, ledger=ledger, gateway=gateway)
    try:
        listener.start()
        config.socket_path.unlink()
        with pytest.raises(PermissionError, match="endpoint disappeared"):
            listener.stop()
        listener.stop()
        assert not config.socket_path.exists()
    finally:
        listener.stop()
        ledger.close()


def test_unsupported_endpoint_limits_refuse_before_bind(tmp_path: Path) -> None:
    """Invalid kernel path or timeout cannot create an endpoint or alter the ledger."""
    config, ledger, gateway = boundary(tmp_path)
    long_socket = config.socket_path.parent / ("s" * 100)
    assert len(os.fsencode(str(long_socket))) >= 108
    cases = (
        ({"socket_path": long_socket}, "endpoint limit"),
        ({"transfer_timeout_seconds": float(TIMEOUT_MAX) * 2}, "platform limit"),
    )
    try:
        for change, message in cases:
            candidate = StorageBoundaryConfiguration.model_validate(config.model_dump() | change)
            listener = StorageRecordListener(candidate, ledger=ledger, gateway=gateway)
            with pytest.raises(ValueError, match=message):
                listener.start()
            assert not candidate.socket_path.exists()
    finally:
        ledger.close()


def test_serve_before_start_refuses_without_creating_endpoint(tmp_path: Path) -> None:
    """An unstarted service cannot accept work or create a socket by serving."""
    config, ledger, gateway = boundary(tmp_path)
    listener = StorageRecordListener(config, ledger=ledger, gateway=gateway)
    try:
        with pytest.raises(RuntimeError, match="not started"):
            listener.serve_once()
        assert not config.socket_path.exists()
    finally:
        ledger.close()
