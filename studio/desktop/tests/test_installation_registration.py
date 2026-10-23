# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual freedesktop application launch

"""Exercise native menu launch and preserve occupied registration paths."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from sc_neurocore.studio.desktop import DesktopStartRefused
from sc_neurocore.studio.desktop import installation as installation_module
from sc_neurocore.studio.desktop.installation import DesktopInstallation
from sc_neurocore.studio.desktop.installation_registration import DesktopRegistration

pytest_plugins = ["studio.desktop.tests.installation_fixtures"]


@pytest.mark.parametrize("slot", [0, 1, 2], ids=["launcher", "manager", "menu"])
def test_install_refuses_launch_fifo_without_waiting_for_a_writer(
    installation: DesktopInstallation,
    installation_environment: dict[str, str],
    desktop_bundles: tuple[Path, Path],
    tmp_path: Path,
    slot: int,
) -> None:
    """Reject real occupied launch paths through installation in a bounded child.

    Parameters
    ----------
    installation : DesktopInstallation
        Fresh actual user profile whose launcher paths are occupied by this case.
    installation_environment : dict of str to str
        Same private HOME/XDG and source environment used by the child.
    desktop_bundles : tuple[pathlib.Path, pathlib.Path]
        Complete real runtime verified by the public installation operation.
    tmp_path : pathlib.Path
        Child working directory outside the application payload.
    slot : int
        Application command, management command or freedesktop entry index.
    """
    path = installation.registration.paths[slot]
    path.parent.mkdir(parents=True, exist_ok=True)
    os.mkfifo(path)
    project = installation.profile.data / "projects" / "retained.json"
    project.parent.mkdir(parents=True, exist_ok=True)
    original = b'{"state":{"seed":"9223372036854775807"}}'
    project.write_bytes(original)
    code = """
from pathlib import Path
import sys
from sc_neurocore.studio.desktop import DesktopStartRefused
from sc_neurocore.studio.desktop.installation import DesktopInstallation
try:
    DesktopInstallation().install(Path(sys.argv[1]))
except DesktopStartRefused as refusal:
    print(str(refusal))
    raise SystemExit(4)
"""
    source = Path(installation_module.__file__).resolve().parents[3]
    child_environment = dict(installation_environment)
    child_environment["PYTHONPATH"] = (
        str(source) + os.pathsep + child_environment.get("PYTHONPATH", "")
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(desktop_bundles[0])],
        env=child_environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 4, result.stdout + result.stderr
    assert "launch file must be a regular file" in result.stdout
    assert stat.S_ISFIFO(path.lstat().st_mode)
    assert project.read_bytes() == original
    assert not installation.root.exists()
    assert all(not other.exists() for other in installation.registration.paths if other != path)


def test_owned_registration_round_trip_preserves_data_and_foreign_changes(
    installation: DesktopInstallation,
    installation_environment: dict[str, str],
    desktop_bundles: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """Exercise registration, refusal and removal against an actual installed runtime.

    Parameters
    ----------
    installation : DesktopInstallation
        Lifecycle owner in a private user profile.
    installation_environment : dict of str to str
        Private profile inherited by the real management command.
    desktop_bundles : tuple[pathlib.Path, pathlib.Path]
        Complete real runtime installed before checking its generated entry points.
    tmp_path : pathlib.Path
        Working directory and retained foreign-file namespace.
    """
    manifest = installation.install(desktop_bundles[0])
    registration = installation.registration
    files = {path: path.read_bytes() for path in registration.paths}
    parents = {path.parent for path in files}
    project = installation.profile.data / "projects" / "registration.json"
    project.parent.mkdir(parents=True, exist_ok=True)
    project.write_bytes(b'{"seed":"9223372036854775807"}')
    project_bytes = project.read_bytes()
    for path, mode in zip(registration.paths, (0o755, 0o755, 0o644), strict=True):
        assert stat.S_IMODE(path.stat().st_mode) == mode
    for command in (
        [str(registration.paths[1]), "--help"],
        ["desktop-file-validate", str(registration.paths[2])],
    ):
        result = subprocess.run(
            command,
            env=installation_environment,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        if command[0] != "desktop-file-validate":
            assert "rollback" in result.stdout and "uninstall" in result.stdout
    registration.verify(managed=True)
    with pytest.raises(DesktopStartRefused, match="existing launch file"):
        registration.register(managed=False)
    registration.register(managed=True)
    assert {path: path.read_bytes() for path in files} == files

    operations: tuple[Callable[[], None], ...] = (
        lambda: registration.register(managed=True),
        registration.remove,
    )
    for path, original in files.items():
        altered = original + b"# retained user edit\n"
        path.write_bytes(altered)
        try:
            for operation in operations:
                with pytest.raises(DesktopStartRefused, match="existing launch file"):
                    operation()
                assert path.read_bytes() == altered
                assert all(
                    other.read_bytes() == data for other, data in files.items() if other != path
                )
        finally:
            path.write_bytes(original)

        external = tmp_path / "retained-command"
        external.write_bytes(original)
        path.unlink()
        path.symlink_to(external)
        try:
            with pytest.raises(DesktopStartRefused, match="launch symlink"):
                registration.remove()
            assert path.is_symlink() and external.read_bytes() == original
        finally:
            path.unlink()
            path.write_bytes(original)

    missing = registration.paths[1]
    missing.unlink()
    registration.verify(managed=True)
    registration.remove()
    registration.remove()
    assert all(not path.exists() for path in files)
    assert all(parent.is_dir() for parent in parents)
    assert project.read_bytes() == project_bytes
    assert (installation.root / "current").resolve().name == manifest.build_id
    registration.register(managed=True)
    assert {path: path.read_bytes() for path in files} == files
    installation.uninstall()
    assert project.read_bytes() == project_bytes
    assert all(parent.is_dir() for parent in parents)


def test_registration_failure_leaves_owned_files_without_temporary_residue(
    installation: DesktopInstallation,
) -> None:
    """Use actual directory permissions to refuse a write without partial files.

    Parameters
    ----------
    installation : DesktopInstallation
        Private user namespace for the launch registration API.
    """
    registration = installation.registration
    registration.register(managed=False)
    directory = registration.paths[0].parent
    files = {path: path.read_bytes() for path in registration.paths}
    contents = set(directory.iterdir())
    mode = stat.S_IMODE(directory.stat().st_mode)
    directory.chmod(0o555)
    try:
        with pytest.raises(PermissionError):
            registration.register(managed=True)
        assert {path: path.read_bytes() for path in files} == files
        assert set(directory.iterdir()) == contents
    finally:
        directory.chmod(mode)
        registration.remove()
    assert directory.is_dir()


@pytest.mark.parametrize("separator", ["\n", "\r", "\t"], ids=["newline", "return", "tab"])
def test_registration_refuses_control_characters_before_creating_launchers(
    installation: DesktopInstallation,
    tmp_path: Path,
    separator: str,
) -> None:
    """Refuse filesystem-valid roots that cannot be represented by desktop commands.

    Parameters
    ----------
    installation : DesktopInstallation
        Private user profile selecting otherwise valid launch-file locations.
    tmp_path : pathlib.Path
        Parent of an actual directory containing the unsupported character.
    separator : str
        Control character which must not enter the freedesktop Exec line.
    """
    root = tmp_path / ("application" + separator + "root")
    root.mkdir()
    registration = DesktopRegistration(root, installation.profile)
    with pytest.raises(DesktopStartRefused, match="cannot be a desktop command"):
        registration.register(managed=False)
    assert not list(root.iterdir())
    assert all(not path.exists() for path in installation.registration.paths)


def test_oversized_foreign_launcher_is_refused_with_bounded_memory(
    installation: DesktopInstallation,
    installation_environment: dict[str, str],
    tmp_path: Path,
) -> None:
    """Refuse an actual oversized launch file under a smaller child memory allowance.

    Parameters
    ----------
    installation : DesktopInstallation
        Unmanaged application namespace whose command is owned by the user.
    installation_environment : dict of str to str
        Private profile inherited by the real source-backed child.
    tmp_path : pathlib.Path
        Child working directory outside the retained user command.
    """
    path = installation.registration.paths[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(b"user-owned command\n")
        stream.truncate(16 * 1024 * 1024)
    original = hashlib.sha256(path.read_bytes()).hexdigest()
    source = Path(installation_module.__file__).resolve().parents[3]
    child_environment = dict(installation_environment)
    child_environment["PYTHONPATH"] = (
        str(source) + os.pathsep + child_environment.get("PYTHONPATH", "")
    )
    code = """
import json
import os
from pathlib import Path
import resource
from sc_neurocore.studio.desktop import DesktopStartRefused
from sc_neurocore.studio.desktop.installation import DesktopInstallation
installation = DesktopInstallation()
pages = int(Path('/proc/self/statm').read_text().split()[0])
allowance = 8 * 1024 * 1024
limit = pages * os.sysconf('SC_PAGE_SIZE') + allowance
resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
print(json.dumps({'virtual_memory_limit': limit, 'extra_virtual_bytes': allowance}), flush=True)
try:
    installation.registration.verify(managed=False)
except DesktopStartRefused as refusal:
    print(str(refusal))
    raise SystemExit(4)
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        env=child_environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 4, result.stdout + result.stderr
    assert "existing launch file" in result.stdout
    observation = json.loads(result.stdout.splitlines()[0])
    assert observation["extra_virtual_bytes"] < path.stat().st_size
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original
    assert not installation.root.exists()
