# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — One API process per identity store, and a private identity file

"""The single-user lab deployment's boundary, held against real processes and files.

Sessions and login throttles live in the API process, so a second process on the
same identity store is refused. The identity file holds credential hashes, so it
is owner-only: a file another account owns is refused, one left readable by
others is narrowed, and every write is owner-only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform.api_process_lock import (
    StudioApiProcessConflict,
    api_lock_path,
    hold_identity_store,
)
from sc_neurocore.studio.platform.identity import (
    add_studio_browser_user_record,
    load_studio_identity_store,
)
from sc_neurocore.studio.platform.settings import StudioRuntimeSettings

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX file modes and owners")


def _identity(path: Path, mode: int = 0o600) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "sc-neurocore.studio.identity.v1",
                "service_accounts": [
                    {
                        "active": True,
                        "principal_id": "svc-admin",
                        "roles": ["studio.admin"],
                        "token_sha256": hashlib.sha256(b"service-token").hexdigest(),
                    }
                ],
                "browser_users": [],
            }
        ),
        encoding="utf-8",
    )
    path.chmod(mode)
    return path


def _settings(identity: Path, tmp_path: Path) -> StudioRuntimeSettings:
    return StudioRuntimeSettings(
        allow_header_principal=False,
        audit_log_path=str(tmp_path / "audit.jsonl"),
        enforce_route_policies=True,
        identity_file_path=str(identity),
    )


_HOLDER = """
import sys
from pathlib import Path
from sc_neurocore.studio.platform.api_process_lock import hold_identity_store
hold_identity_store(Path(sys.argv[1]))
print("held", flush=True)
sys.stdin.read()
"""


def _holder(identity: Path) -> subprocess.Popen[str]:
    process = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(identity)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "held"
    return process


def _release(process: subprocess.Popen[str]) -> None:
    assert process.stdin is not None
    process.stdin.close()
    assert process.wait(timeout=30) == 0


class TestOneApiProcess:
    def test_the_holding_process_may_open_its_store_again(self, tmp_path: Path) -> None:
        identity = _identity(tmp_path / "identity.json")
        assert hold_identity_store(identity) == api_lock_path(identity)
        assert hold_identity_store(identity) == api_lock_path(identity)
        create_app(_settings(identity, tmp_path))
        create_app(_settings(identity, tmp_path))

    def test_a_second_process_on_the_same_store_is_refused(self, tmp_path: Path) -> None:
        identity = _identity(tmp_path / "identity.json")
        holder = _holder(identity)
        try:
            with pytest.raises(
                StudioApiProcessConflict, match="another Studio API process serves identity.json"
            ):
                create_app(_settings(identity, tmp_path))
        finally:
            _release(holder)

    def test_the_store_is_free_once_its_process_has_ended(self, tmp_path: Path) -> None:
        identity = _identity(tmp_path / "identity.json")
        _release(_holder(identity))
        create_app(_settings(identity, tmp_path))

    def test_separate_stores_are_served_by_separate_processes(self, tmp_path: Path) -> None:
        holder = _holder(_identity(tmp_path / "other.json"))
        try:
            create_app(_settings(_identity(tmp_path / "identity.json"), tmp_path))
        finally:
            _release(holder)

    def test_an_app_without_an_identity_store_holds_nothing(self, tmp_path: Path) -> None:
        create_app(StudioRuntimeSettings())
        assert not list(tmp_path.glob("*.api-lock"))


@posix_only
class TestPrivateIdentityFile:
    def test_a_store_readable_by_others_is_narrowed_before_it_is_read(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        identity = _identity(tmp_path / "identity.json", mode=0o644)
        with caplog.at_level(logging.WARNING, logger="sc_neurocore.studio.identity"):
            store = load_studio_identity_store(identity)
        assert store.service_accounts[0].principal_id == "svc-admin"
        assert identity.stat().st_mode & 0o777 == 0o600
        assert "readable beyond its owner (mode 644); narrowed to 600" in caplog.text

    def test_an_owner_only_store_is_left_as_it_is(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        identity = _identity(tmp_path / "identity.json", mode=0o400)
        with caplog.at_level(logging.WARNING, logger="sc_neurocore.studio.identity"):
            load_studio_identity_store(identity)
        assert identity.stat().st_mode & 0o777 == 0o400
        assert caplog.text == ""

    @pytest.mark.skipif(
        not Path("/etc/passwd").exists() or os.geteuid() == 0,
        reason="needs a file another account owns",
    )
    def test_a_store_another_account_owns_is_refused(self) -> None:
        """``/etc/passwd`` is a real file root owns; its content never matters here."""
        with pytest.raises(ValueError, match="must be owned by the account running Studio"):
            load_studio_identity_store(Path("/etc/passwd"))

    def test_every_write_is_owner_only_whatever_the_file_was(self, tmp_path: Path) -> None:
        identity = _identity(tmp_path / "identity.json", mode=0o644)
        add_studio_browser_user_record(
            identity,
            principal_id="user-operator",
            username="operator",
            password="browser-password-long-enough",
            roles=["studio.viewer"],
        )
        assert identity.stat().st_mode & 0o777 == 0o600
