# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bundled installation command integration

"""Use the real bootstrap and installed command without a developer interpreter."""

from __future__ import annotations

import json
import shutil
import sys
import subprocess
from pathlib import Path

import pytest

from sc_neurocore.studio.desktop.bundle import verify_bundle
from sc_neurocore.studio.desktop.installation import DesktopInstallation

pytest_plugins = ["studio.desktop.tests.installation_fixtures"]


@pytest.mark.parametrize("entry", ["bundle", "source-module"])
def test_bundled_commands_install_upgrade_rollback_and_remove(
    installation: DesktopInstallation,
    installation_environment: dict[str, str],
    desktop_bundles: tuple[Path, Path],
    tmp_path: Path,
    entry: str,
) -> None:
    """Run actual executable commands and remove their own independent interpreter.

    Parameters
    ----------
    installation : DesktopInstallation
        Real lifecycle observer for the disposable user.
    installation_environment : dict of str to str
        Same real HOME/XDG selection inherited by every command.
    desktop_bundles : tuple[pathlib.Path, pathlib.Path]
        Two full runtimes from the same qualified research data family.
    tmp_path : pathlib.Path
        Private process working directory and retained research data.
    entry : str
        Bundled bootstrap and manager, or the actual source module command.
    """
    first = verify_bundle(desktop_bundles[0])
    second = verify_bundle(desktop_bundles[1])
    source_command = [sys.executable, "-I", "-B", "-m", "sc_neurocore.studio.desktop.install"]
    initial_command = (
        [str(desktop_bundles[0] / "install.sh")]
        if entry == "bundle"
        else [*source_command, "install", str(desktop_bundles[0])]
    )
    initial = subprocess.run(
        initial_command,
        env=installation_environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert initial.returncode == 0, initial.stdout + initial.stderr
    assert first.build_id in initial.stdout
    manager = installation.registration.paths[1]
    management_command = [str(manager)] if entry == "bundle" else source_command
    python = installation.root / "current/runtime/bin/python3.12"
    probe = subprocess.run(
        [
            str(python),
            "-I",
            "-B",
            "-c",
            "import json,sys,sc_neurocore.studio.desktop.install as module; "
            "print(json.dumps({'isolated':sys.flags.isolated,'prefix':sys.prefix,'module':module.__file__}))",
        ],
        env=installation_environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
    identity = json.loads(probe.stdout)
    selected = installation.root / "versions" / first.build_id
    assert identity["isolated"] == 1
    assert Path(identity["prefix"]).resolve() == selected / "runtime"
    assert Path(identity["module"]).resolve().is_relative_to(selected / "runtime")
    project = installation.profile.data / "projects" / "retained.json"
    project.parent.mkdir()
    project.write_bytes(b'{"seed":"9223372036854775807"}')
    for arguments, selected_id in [
        (["install", str(desktop_bundles[1])], second.build_id),
        (["rollback", first.build_id], first.build_id),
    ]:
        result = subprocess.run(
            [*management_command, *arguments],
            env=installation_environment,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert selected_id in result.stdout
        assert (installation.root / "current").resolve().name == selected_id
    removed = subprocess.run(
        [*management_command, "uninstall"],
        env=installation_environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert removed.returncode == 0, removed.stdout + removed.stderr
    assert not installation.root.exists()
    assert all(not path.exists() for path in installation.registration.paths)


def test_install_callable_refuses_removal_of_an_uninstalled_profile(
    installation: DesktopInstallation,
    installation_environment: dict[str, str],
    tmp_path: Path,
) -> None:
    """Use the public callable in an isolated real child and retain unrelated data.

    Parameters
    ----------
    installation : DesktopInstallation
        Fresh profile whose application has never been installed.
    installation_environment : dict of str to str
        Real private HOME/XDG environment inherited by the callable consumer.
    tmp_path : pathlib.Path
        Process working directory outside the source and user data.
    """
    installation.profile.prepare()
    project = installation.profile.data / "retained.json"
    original = b'{"seed":"9223372036854775807"}'
    project.write_bytes(original)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            "from sc_neurocore.studio.desktop.install import main; raise SystemExit(main())",
            "uninstall",
        ],
        env=installation_environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "SNN Studio is not installed for this profile." in result.stderr
    assert result.stdout == ""
    assert project.read_bytes() == original
    assert not installation.root.exists()
    assert all(not path.exists() for path in installation.registration.paths)
    assert project.read_bytes() == b'{"seed":"9223372036854775807"}'


def test_install_command_refuses_an_unsupported_kernel_machine(
    desktop_bundles: tuple[Path, Path],
    installation: DesktopInstallation,
    installation_environment: dict[str, str],
    tmp_path: Path,
) -> None:
    """Refuse the real i686 kernel personality before installing any application.

    Parameters
    ----------
    desktop_bundles : tuple[pathlib.Path, pathlib.Path]
        Full runtime presented to the actual module installation command.
    installation : DesktopInstallation
        Observer of the same disposable profile used by the child.
    installation_environment : dict of str to str
        Actual private HOME/XDG environment, with normal source imports.
    tmp_path : pathlib.Path
        Child working directory independent of the package input.

    Notes
    -----
    Linux setarch changes only the child's kernel-reported machine. This tests
    real platform admission, not execution on a physical 32-bit CPU or acceptance
    of an installed distribution on another operating system.
    """
    command = shutil.which("setarch")
    assert command is not None, "The Linux desktop qualification requires util-linux setarch."
    result = subprocess.run(
        [
            command,
            "linux32",
            sys.executable,
            "-B",
            "-m",
            "sc_neurocore.studio.desktop.install",
            "install",
            str(desktop_bundles[0]),
        ],
        cwd=tmp_path,
        env=installation_environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "This desktop bundle requires Linux x86_64." in result.stderr
    assert not installation.root.exists()
    assert all(not path.exists() for path in installation.registration.paths)
