# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — worker launcher configuration and startup

"""Operator configuration, startup refusals and endpoint ownership are strict."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs_process_protocol import _process_worker_environment
from sc_neurocore.studio.platform.storage_launcher_client import (
    new_launcher_request,
)
from sc_neurocore.studio.platform.storage_launcher_configuration import (
    CONFIGURATION_MAX_BYTES,
    LauncherConfiguration,
    load_launcher_configuration,
)
from sc_neurocore.studio.platform.storage_worker_launcher import WorkerLauncher
from tests.studio_storage_launcher_support import *


@pytest.fixture
def base() -> Iterator[Path]:
    """Short private base directory so every socket path fits the Unix limit."""
    with launcher_base() as path:
        yield path


@pytest.mark.parametrize("fault", ["existing", "shared-parent", "foreign-worker"])
def test_launcher_refuses_unsafe_startup(base: Path, fault: str) -> None:
    """Occupied endpoints, shared socket parents and unattainable identities refuse."""
    changes: dict[str, object] = {}
    if fault == "existing":
        (base / "sock" / "launcher.sock").write_text("occupied")
    elif fault == "shared-parent":
        (base / "sock").chmod(0o770)
    else:
        changes["worker_uid"] = os.getuid() + 1
    config_path = base / "launcher.json"
    config_path.write_text(json.dumps(configuration(base, **changes)))
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sc_neurocore.studio.platform.storage_launcher_service",
            "--configuration",
            str(config_path),
        ],
        env=_process_worker_environment(),
        capture_output=True,
        timeout=60.0,
        check=False,
    )
    assert completed.returncode != 0
    assert completed.stdout == b""
    expected = "FileExistsError" if fault == "existing" else "PermissionError"
    assert expected in completed.stderr.decode()
    if fault == "existing":
        assert (base / "sock" / "launcher.sock").read_text() == "occupied"


@pytest.mark.parametrize(
    "content,match",
    [
        (b"", "byte limit"),
        (b" " * (CONFIGURATION_MAX_BYTES + 1), "byte limit"),
        (b"\xff", "invalid launcher configuration JSON"),
        (b'{"api_uid":1,"api_uid":1}', "duplicate launcher configuration field"),
    ],
)
def test_configuration_file_refuses_malformed_content(
    base: Path, content: bytes, match: str
) -> None:
    """The operator file is strict, bounded and duplicate-free."""
    path = base / "launcher.json"
    path.write_bytes(content)
    with pytest.raises(ValueError, match=match):
        load_launcher_configuration(path)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"spool_root": "spool"}, "absolute and normalised"),
        ({"python_executable": "/usr/bin/../bin/python3"}, "absolute and normalised"),
        ({"python_path": []}, "import root"),
        ({"max_workers": 0}, "max_workers"),
        ({"transfer_timeout_seconds": 0}, "transfer_timeout_seconds"),
        ({"worker_uid": 0}, "worker_uid"),
        ({"command": "/bin/sh"}, "Extra inputs"),
    ],
)
def test_configuration_fields_are_validated(
    base: Path, changes: dict[str, object], match: str
) -> None:
    """Paths, budgets and identities are checked; unknown fields refuse."""
    path = base / "launcher.json"
    path.write_text(json.dumps(configuration(base, **changes)))
    with pytest.raises(ValueError, match=match):
        load_launcher_configuration(path)


def test_foreign_spool_owner_refuses_before_any_spawn(base: Path) -> None:
    """A generation directory not owned by the API identity starts nothing.

    ``handle`` refuses before spawning, so this case runs in-process without
    starting the service or changing the runner's process role.
    """
    config = LauncherConfiguration.model_validate_json(
        json.dumps(configuration(base, api_uid=os.getuid() + 1)), strict=True
    )
    prepare(base / "spool", JOB_A, GEN_A)
    service = WorkerLauncher(config)
    response = service.handle(new_launcher_request("launch", job_id=JOB_A, generation=GEN_A))
    assert (response.state, response.reason) == ("refused", "spool")
    with pytest.raises(RuntimeError, match="not started"):
        service.serve_once()
    service.stop()
    assert service.privileged is False


def test_unopened_endpoint_close_touches_nothing(base: Path) -> None:
    """Closing an endpoint that never bound leaves any existing entry alone."""
    from sc_neurocore.studio.platform.storage_launcher_endpoint import LauncherEndpoint

    planted = base / "sock" / "launcher.sock"
    planted.write_text("foreign")
    LauncherEndpoint(planted).close()
    assert planted.read_text() == "foreign"


def test_service_entry_refuses_missing_configuration(base: Path) -> None:
    """The service stops before installing handlers when its configuration is absent."""
    from sc_neurocore.studio.platform.storage_launcher_service import main

    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(FileNotFoundError):
        main(["--configuration", str(base / "absent.json")])
    assert signal.getsignal(signal.SIGTERM) is before


@pytest.mark.parametrize("fault", ["removed", "replaced"])
def test_shutdown_never_unlinks_a_foreign_endpoint(base: Path, fault: str) -> None:
    """Shutdown removes only the launcher's own unchanged socket inode."""
    running = start(base)
    endpoint = base / "sock" / "launcher.sock"
    endpoint.unlink()
    if fault == "replaced":
        endpoint.write_text("replacement")
    shutdown(running)
    assert running.process.returncode == 0
    if fault == "replaced":
        assert endpoint.read_text() == "replacement"
    else:
        assert not endpoint.exists()
