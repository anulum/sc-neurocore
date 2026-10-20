# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Configured project persistence and recovery integration

"""Exercise a selected project namespace across real API and CLI processes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sc_neurocore.studio.project_paths import (
    PROJECT_ROOT_ENV,
    StudioProjectRootRefused,
    resolve_studio_project_root,
)

ROOT = Path(__file__).resolve().parents[1]
API_DRIVER = '''# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public project API process driver

"""Perform project requests through the actual application in a disposable user profile."""
import json
import sys
from fastapi.testclient import TestClient
from sc_neurocore.studio.app import create_app

requests = json.load(sys.stdin)
replies = []
with TestClient(
    create_app(),
    base_url="http://127.0.0.1:8001",
    headers={"Origin": "http://127.0.0.1:8001"},
) as client:
    for request in requests:
        response = client.request(
            request["method"], request["url"], json=request.get("body")
        )
        replies.append({"status": response.status_code, "json": response.json()})
print(json.dumps(replies))
'''


def _environment(directory: Path, project_root: str | None) -> dict[str, str]:
    home = directory / "user-profile"
    temporary = directory / "tmp"
    home.mkdir(exist_ok=True)
    temporary.mkdir(exist_ok=True)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SC_NEUROCORE_STUDIO_")
    }
    env.update(
        HOME=str(home),
        TMPDIR=str(temporary),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=os.pathsep.join((str(ROOT / "src"), str(ROOT))),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        SC_NEUROCORE_STUDIO_JOB_ROOT=str(directory / "jobs"),
        SC_NEUROCORE_STUDIO_AUDIT_LOG_PATH=str(directory / "audit.jsonl"),
        SC_NEUROCORE_STUDIO_BENCHMARK_DATABANK=str(directory / "benchmark.jsonl"),
    )
    if project_root is not None:
        env[PROJECT_ROOT_ENV] = project_root
    return env


def _api(
    directory: Path, project_root: Path | None, requests: list[dict[str, object]]
) -> list[dict[str, object]]:
    driver = directory / "project-api.py"
    driver.write_text(API_DRIVER)
    result = subprocess.run(
        [sys.executable, str(driver)],
        input=json.dumps(requests),
        cwd=ROOT,
        env=_environment(directory, None if project_root is None else str(project_root)),
        capture_output=True,
        text=True,
        timeout=40,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    response: object = json.loads(result.stdout)
    assert isinstance(response, list)
    replies: list[dict[str, object]] = []
    for row in response:
        assert isinstance(row, dict)
        replies.append(row)
    return replies


def test_project_namespace_survives_restart_and_matches_cli_backup(tmp_path: Path) -> None:
    """Keep configured, default and second namespaces distinct across processes.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Disposable user profile, runtime and project directories on the test disk.
    """
    first = tmp_path / "data" / "first"
    second = tmp_path / "data" / "second"
    state = {"seed": str(2**63 - 1), "equations": ["dv/dt = -v / 10"]}
    create: list[dict[str, object]] = [
        {
            "method": "POST",
            "url": "/api/project/save",
            "body": {"name": "experiment", "state": state},
        }
    ]
    saved = _api(tmp_path, first, create)
    assert saved[0]["status"] == 200, saved
    loaded = _api(tmp_path, first, [{"method": "GET", "url": "/api/project/load/experiment"}])
    assert loaded[0]["status"] == 200
    document = loaded[0]["json"]
    assert isinstance(document, dict)
    assert document["state"] == state
    assert any(first.iterdir())
    assert not (tmp_path / "user-profile" / ".sc-neurocore" / "studio" / "projects").exists()

    other = _api(tmp_path, second, [{"method": "GET", "url": "/api/project/list"}])
    assert other == [{"status": 200, "json": []}]
    original = _api(tmp_path, first, [{"method": "GET", "url": "/api/project/list"}])
    rows = original[0]["json"]
    assert isinstance(rows, list)
    assert [row["name"] for row in rows] == ["experiment"]

    backup = subprocess.run(
        [sys.executable, "-m", "sc_neurocore.cli", "studio-backup-plan", "--include-local-paths"],
        cwd=ROOT,
        env=_environment(tmp_path, str(first)),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert backup.returncode == 0, backup.stdout + backup.stderr
    plan = json.loads(backup.stdout)
    project_item = next(item for item in plan["items"] if item["item_id"] == "project_workspace")
    assert project_item["local_path"] == str(first.resolve())
    assert project_item["exists"] is True
    assert project_item["source_label"] == PROJECT_ROOT_ENV
    public = subprocess.run(
        [sys.executable, "-m", "sc_neurocore.cli", "studio-backup-plan"],
        cwd=ROOT,
        env=_environment(tmp_path, str(first)),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert public.returncode == 0, public.stdout + public.stderr
    assert str(first) not in public.stdout
    public_plan = json.loads(public.stdout)
    assert all("local_path" not in item for item in public_plan["items"])


def test_legacy_project_namespace_remains_the_default(tmp_path: Path) -> None:
    """Keep existing user projects readable without opting into another root.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Disposable home profile and runtime directories.
    """
    state = {"seed": "9007199254740993"}
    replies = _api(
        tmp_path,
        None,
        [
            {
                "method": "POST",
                "url": "/api/project/save",
                "body": {"name": "legacy", "state": state},
            },
            {"method": "GET", "url": "/api/project/load/legacy"},
        ],
    )
    assert [reply["status"] for reply in replies] == [200, 200]
    document = replies[1]["json"]
    assert isinstance(document, dict)
    assert document["state"] == state
    assert (tmp_path / "user-profile" / ".sc-neurocore" / "studio" / "projects").is_dir()


@pytest.mark.parametrize("configured", ["", "   ", "relative/projects", "~/projects"])
def test_cli_rejects_invalid_project_roots_before_startup(tmp_path: Path, configured: str) -> None:
    """Refuse unusable operator configuration before creating runtime state.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Disposable user profile and runtime directories.
    configured : str
        Empty or relative project-root configuration.
    """
    result = subprocess.run(
        [sys.executable, "-m", "sc_neurocore.cli", "studio"],
        cwd=ROOT,
        env=_environment(tmp_path, configured),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Studio project root must be a nonempty absolute path." in result.stdout
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "jobs").exists()
    assert not (tmp_path / "audit.jsonl").exists()


def test_project_root_resolution_is_nonmutating_and_canonical(tmp_path: Path) -> None:
    """Resolve configured aliases and retain a caller's legacy location.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory holding a real target and a filesystem alias.
    """
    target = tmp_path / "data"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    assert resolve_studio_project_root(environ={PROJECT_ROOT_ENV: str(alias)}) == target
    legacy = tmp_path / "legacy"
    assert resolve_studio_project_root(environ={}, legacy_root=legacy) == legacy
    assert not legacy.exists()


def test_project_root_refuses_files_and_unresolvable_paths(tmp_path: Path) -> None:
    """Refuse file roots, invalid path bytes and a genuine cyclic filesystem alias.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory holding a regular file and a cyclic alias.
    """
    file = tmp_path / "file"
    file.write_text("existing content")
    with pytest.raises(StudioProjectRootRefused, match="must name a directory"):
        resolve_studio_project_root(environ={PROJECT_ROOT_ENV: str(file)})
    with pytest.raises(StudioProjectRootRefused, match="could not be resolved"):
        resolve_studio_project_root(environ={PROJECT_ROOT_ENV: str(tmp_path) + "/bad\0path"})
    alias = tmp_path / "cycle"
    alias.symlink_to(alias)
    with pytest.raises(StudioProjectRootRefused, match="could not be resolved"):
        resolve_studio_project_root(environ={PROJECT_ROOT_ENV: str(alias)})
