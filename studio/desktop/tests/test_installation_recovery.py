# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Interrupted real desktop installation recovery

"""Recover an actual interrupted bundled installer without losing research data."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from sc_neurocore.studio.desktop import DesktopStartRefused
from sc_neurocore.studio.desktop.bundle import create_bundle_manifest, verify_bundle
from sc_neurocore.studio.desktop.configuration import DesktopConfiguration, write_configuration
from sc_neurocore.studio.desktop.installation import DesktopInstallation
from sc_neurocore.studio.workspace_schema import write_atomic

pytest_plugins = ["studio.desktop.tests.installation_fixtures"]


def test_reinstall_recovers_its_actual_interrupted_payload_copy(
    installation: DesktopInstallation,
    installation_environment: dict[str, str],
    desktop_bundles: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """Interrupt the owned real CLI mid-copy and recover through the public API.

    Parameters
    ----------
    installation : DesktopInstallation
        Fresh lifecycle controller using the same disposable user as the child.
    installation_environment : dict of str to str
        Private profile inherited by the actual isolated bundled interpreter.
    desktop_bundles : tuple[pathlib.Path, pathlib.Path]
        Complete real runtime inputs; the first is independently copied twice.
    tmp_path : pathlib.Path
        Retained interruption observations and child output.
    """
    installation.profile.prepare()
    projects = installation.profile.data / "projects"
    projects.mkdir()
    project = projects / "retained.json"
    project.write_bytes(b'{"seed":"9223372036854775807"}')
    configuration = installation.profile.config / "desktop.json"
    write_configuration(configuration, DesktopConfiguration(18031, projects))
    cache = installation.profile.cache / "retained.bin"
    cache.write_bytes(b"retained research cache")
    before = {path: path.read_bytes() for path in (project, configuration, cache)}
    source = desktop_bundles[0]
    manifest = verify_bundle(source)
    expected_files = [entry.path for entry in manifest.entries if entry.kind == "file"]
    command = [
        str(source / "runtime/bin/python3.12"),
        "-I",
        "-B",
        "-m",
        "sc_neurocore.studio.desktop.install",
        "install",
        str(source),
    ]
    environment = {
        key: value
        for key, value in installation_environment.items()
        if key not in {"PYTHONPATH", "PYTHONHOME"} and not key.startswith("SC_NEUROCORE_STUDIO_")
    }
    stage: Path | None = None
    copied_bytes = 0
    copied_path: str | None = None
    with (tmp_path / "interrupted-installer.txt").open("wb") as output:
        process = subprocess.Popen(
            command, env=environment, cwd=tmp_path, stdout=output, stderr=output
        )
        try:
            deadline = time.monotonic() + 120
            while process.poll() is None and time.monotonic() < deadline:
                for candidate in (installation.root / "versions").glob(".stage-*"):
                    payload = candidate / "payload"
                    if not (candidate / "owner.json").is_file():
                        continue
                    for copied in payload.rglob("*"):
                        if copied.is_file() and not copied.is_symlink() and copied.stat().st_size:
                            copied_bytes = copied.stat().st_size
                            copied_path = copied.relative_to(payload).as_posix()
                            stage = candidate
                            break
                    if stage is not None:
                        break
                if stage is not None:
                    break
                time.sleep(0.02)
            assert stage is not None, (tmp_path / "interrupted-installer.txt").read_text()
            assert process.poll() is None
            assert not (installation.root / "current").exists()
            with pytest.raises(DesktopStartRefused, match="Close Studio"):
                installation.uninstall()
            process.kill()
            assert process.wait(timeout=10) == -signal.SIGKILL
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
    assert stage.exists()
    missing_after_kill = [
        name for name in expected_files if not (stage / "payload" / name).is_file()
    ]
    assert missing_after_kill, "The process must be interrupted before its payload is complete."
    assert {path: path.read_bytes() for path in before} == before
    recovered = installation.install(source)
    assert recovered == manifest
    assert not stage.exists()
    assert not list((installation.root / "versions").glob(".stage-*"))
    assert (installation.root / "current").readlink() == Path("versions") / recovered.build_id
    assert (installation.root / "current/runtime/bin/python3.12").stat().st_ino != (
        source / "runtime/bin/python3.12"
    ).stat().st_ino
    assert {path: path.read_bytes() for path in before} == before
    (tmp_path / "interruption-observation.json").write_text(
        json.dumps(
            {
                "command": command,
                "owned_pid": process.pid,
                "exit": process.returncode,
                "interrupted_stage": str(stage),
                "observed_copied_path": copied_path,
                "observed_copied_bytes": copied_bytes,
                "payload_files_missing_after_kill": len(missing_after_kill),
                "recovered_build_id": recovered.build_id,
                "research_files_preserved": True,
            },
            indent=2,
        )
        + "\n"
    )


def test_install_refuses_a_resealed_staged_manifest_during_its_actual_payload_copy(
    installation: DesktopInstallation,
    installation_environment: dict[str, str],
    desktop_bundles: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """Preserve research when the staged copy acquires a different valid identity.

    Parameters
    ----------
    installation : DesktopInstallation
        Fresh profile observed independently of the actual installer child.
    installation_environment : dict of str to str
        Real HOME/XDG environment inherited by the public module command.
    desktop_bundles : tuple[pathlib.Path, pathlib.Path]
        Complete runtime used to generate a real alternate seal for the staged copy.
    tmp_path : pathlib.Path
        Real child logs and source-change observation record.
    """
    installation.profile.prepare()
    project = installation.profile.data / "retained.json"
    research = b'{"seed":"9223372036854775807"}'
    project.write_bytes(research)
    source = desktop_bundles[0]
    original_seal = (source / "bundle.json").read_text()
    original = verify_bundle(source)
    command = [
        sys.executable,
        "-I",
        "-B",
        "-m",
        "sc_neurocore.studio.desktop.install",
        "install",
        str(source),
    ]
    stage: Path | None = None
    copied_path: str | None = None
    with (tmp_path / "changed-source-installer.txt").open("wb") as output:
        process = subprocess.Popen(
            command, env=installation_environment, cwd=tmp_path, stdout=output, stderr=output
        )
        try:
            deadline = time.monotonic() + 120
            while process.poll() is None and time.monotonic() < deadline:
                for candidate in (installation.root / "versions").glob(".stage-*"):
                    payload = candidate / "payload"
                    if not (candidate / "owner.json").is_file():
                        continue
                    for copied in payload.rglob("*"):
                        if copied.is_file() and not copied.is_symlink() and copied.stat().st_size:
                            stage = candidate
                            copied_path = copied.relative_to(payload).as_posix()
                            break
                    if stage is not None:
                        break
                if stage is not None:
                    break
                time.sleep(0.02)
            assert stage is not None, (tmp_path / "changed-source-installer.txt").read_text()
            process.send_signal(signal.SIGSTOP)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                state = (Path("/proc") / str(process.pid) / "stat").read_text()
                if state.rsplit(")", 1)[1].split()[0] == "T":
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("The actual installer child did not stop during copying.")
            assert any(
                not (stage / "payload" / entry.path).is_file()
                for entry in original.entries
                if entry.kind == "file"
            ), "The actual payload copy must still be incomplete when its seal changes."
            changed = create_bundle_manifest(
                source,
                version="3.16.0+changed.during.copy",
                compatibility_key=original.compatibility_key,
            )
            assert changed.build_id != original.build_id
            assert verify_bundle(source) == changed
            write_atomic(stage / "payload/bundle.json", (source / "bundle.json").read_text())
            process.send_signal(signal.SIGCONT)
            assert process.wait(timeout=180) == 1
            log = (tmp_path / "changed-source-installer.txt").read_text()
            assert "The copied desktop bundle does not match." in log
            assert not stage.exists()
            assert list((installation.root / "versions").iterdir()) == []
            assert not (installation.root / "current").is_symlink()
            assert all(not path.exists() for path in installation.registration.paths)
            assert not (installation.profile.state / "application-data.json").exists()
            assert project.read_bytes() == research
            (tmp_path / "changed-source-observation.json").write_text(
                json.dumps(
                    {
                        "command": command,
                        "owned_pid": process.pid,
                        "copied_path_before_stop": copied_path,
                        "original_build_id": original.build_id,
                        "changed_build_id": changed.build_id,
                        "staged_manifest_replaced": True,
                        "exit": process.returncode,
                        "research_preserved": True,
                        "activation_refused": True,
                    },
                    indent=2,
                )
                + "\n"
            )
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
            write_atomic(source / "bundle.json", original_seal)
    assert verify_bundle(source) == original
