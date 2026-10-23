# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete desktop delivery archive integration

"""Export actual complete runtimes and install the bytes consumers extract."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from sc_neurocore.studio.desktop import DesktopStartRefused
from sc_neurocore.studio.desktop.archive import create_bundle_archive
from sc_neurocore.studio.desktop.bundle import create_bundle_manifest, verify_bundle
from sc_neurocore.studio.desktop.installation import DesktopInstallation

pytest_plugins = ["studio.desktop.tests.installation_fixtures"]


@pytest.fixture(scope="module")
def mutable_bundle(
    desktop_bundles: tuple[Path, Path], tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Path]:
    """Copy a complete real input for source-mutation checks without editing it.

    Parameters
    ----------
    desktop_bundles : tuple[pathlib.Path, pathlib.Path]
        Real retained qualification inputs, left byte-identical.
    tmp_path_factory : pytest.TempPathFactory
        Owner of the disposable full input copy.

    Yields
    ------
    pathlib.Path
        Independent complete runtime with unchanged seal and ordinary modes.
    """
    root = tmp_path_factory.mktemp("archive-mutation") / "bundle"
    shutil.copytree(desktop_bundles[0], root, symlinks=True)
    try:
        yield root
    finally:
        shutil.rmtree(root)


def test_archive_is_reproducible_and_its_extracted_runtime_installs(
    desktop_bundles: tuple[Path, Path],
    installation: DesktopInstallation,
    installation_environment: dict[str, str],
    tmp_path: Path,
) -> None:
    """Round-trip every real entry and install through the extracted bootstrap.

    Parameters
    ----------
    desktop_bundles : tuple[pathlib.Path, pathlib.Path]
        Complete real scientific/UI/Qt runtime, with its existing provenance.
    installation : DesktopInstallation
        Actual disposable profile, removed through its normal lifecycle.
    installation_environment : dict of str to str
        Private user/XDG environment shared by all installed commands.
    tmp_path : pathlib.Path
        Archive, extraction and process working directory.
    """
    bundle = desktop_bundles[0]
    manifest = verify_bundle(bundle)
    first, second = tmp_path / "first.tar.gz", tmp_path / "different-name.tar.gz"
    try:
        result = create_bundle_archive(bundle, first)
        command = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "sc_neurocore.studio.desktop.archive",
                str(bundle),
                str(second),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert command.returncode == 0, command.stdout + command.stderr
        with second.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        assert actual == result.sha256
        assert result.build_id == manifest.build_id
        assert result.size_bytes == first.stat().st_size == second.stat().st_size
        assert result.sha256 in command.stdout and manifest.build_id in command.stdout
        extracted = tmp_path / "extracted payload"
        extracted.mkdir()
        with tarfile.open(first, "r:gz") as archive:
            members = archive.getmembers()
            assert len(members) == len(manifest.entries) + 1
            assert all(row.uid == row.gid == row.mtime == 0 for row in members)
            assert all(row.uname == row.gname == "" for row in members)
            # This archive was created above from the verified local bundle;
            # preserve ordinary sealed modes rather than rewriting its contents.
            archive.extractall(extracted, filter="fully_trusted")
        assert verify_bundle(extracted) == manifest
        assert (extracted / "bundle.json").read_bytes() == (bundle / "bundle.json").read_bytes()
        installed = subprocess.run(
            [str(extracted / "install.sh")],
            env=installation_environment,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert installed.returncode == 0, installed.stdout + installed.stderr
        assert manifest.build_id in installed.stdout
        python = installation.root / "current/runtime/bin/python3.12"
        probe = subprocess.run(
            [
                str(python),
                "-I",
                "-B",
                "-c",
                "import json,sys,numpy as np; from importlib.resources import files; "
                "from sc_neurocore.neurons.models import AdExNeuron; "
                "n=AdExNeuron(); trace=[n.step(1.0) for _ in range(20)]; "
                "print(json.dumps({'isolated':sys.flags.isolated,'prefix':sys.prefix,"
                "'finite':bool(np.isfinite(trace).all()),'ui':files('sc_neurocore.studio')."
                "joinpath('frontend_dist/index.html').is_file()}))",
            ],
            env=installation_environment,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert probe.returncode == 0, probe.stdout + probe.stderr
        identity = json.loads(probe.stdout)
        assert identity["isolated"] == 1 and identity["finite"] and identity["ui"]
        assert Path(identity["prefix"]).resolve() == (
            installation.root / "versions" / manifest.build_id / "runtime"
        )
    finally:
        first.unlink(missing_ok=True)
        second.unlink(missing_ok=True)
        if (tmp_path / "extracted payload").exists():
            shutil.rmtree(tmp_path / "extracted payload")


@pytest.mark.parametrize("kind", ["inside", "file", "dangling-link"])
def test_archive_refuses_invalid_output_without_replacing_files(
    desktop_bundles: tuple[Path, Path], tmp_path: Path, kind: str
) -> None:
    """Keep both input seals and foreign destination entries intact.

    Parameters
    ----------
    desktop_bundles : tuple[pathlib.Path, pathlib.Path]
        Complete unchanged input runtime.
    tmp_path : pathlib.Path
        Foreign destination namespace outside the runtime.
    kind : str
        Invalid input-contained path, existing file or broken symbolic link.
    """
    bundle = desktop_bundles[0]
    seal = (bundle / "bundle.json").read_bytes()
    target = bundle / "delivery.tar.gz" if kind == "inside" else tmp_path / "delivery.tar.gz"
    if kind == "file":
        target.write_bytes(b"owner artifact")
    elif kind == "dangling-link":
        target.symlink_to("retained-missing-target")
    with pytest.raises(DesktopStartRefused, match="outside|already exists"):
        create_bundle_archive(bundle, target)
    if kind == "file":
        assert target.read_bytes() == b"owner artifact"
    elif kind == "dangling-link":
        assert target.is_symlink() and os.readlink(target) == "retained-missing-target"
    else:
        assert not target.exists()
    assert (bundle / "bundle.json").read_bytes() == seal
    assert not list(tmp_path.glob(".snn-studio-archive-*"))


@pytest.mark.parametrize("kind", ["unsealed-input", "missing-output-directory"])
def test_archive_command_reports_real_input_and_filesystem_refusals(
    desktop_bundles: tuple[Path, Path], tmp_path: Path, kind: str
) -> None:
    """Use actual missing filesystem inputs through the public module command.

    Parameters
    ----------
    desktop_bundles : tuple[pathlib.Path, pathlib.Path]
        Real input when testing the absent output directory.
    tmp_path : pathlib.Path
        Independent destination where no partial file may appear.
    kind : str
        Existing unsealed directory or absent parent of the output archive.
    """
    bundle = tmp_path / "unsealed" if kind == "unsealed-input" else desktop_bundles[0]
    if kind == "unsealed-input":
        bundle.mkdir()
    target = tmp_path / "missing" / "delivery.tar.gz"
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "sc_neurocore.studio.desktop.archive",
            str(bundle),
            str(target),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout == "" and "Traceback" not in result.stderr
    assert not target.exists() and not target.parent.exists()


@pytest.mark.parametrize(
    "change", ["payload", "manifest-bytes", "manifest-identity", "destination"]
)
def test_archive_never_publishes_a_changed_input_or_clobbers_a_racing_destination(
    mutable_bundle: Path, tmp_path: Path, change: str
) -> None:
    """Stop a real exporter mid-write and alter its actual filesystem boundary.

    Parameters
    ----------
    mutable_bundle : pathlib.Path
        Complete independent real-runtime copy, restored after every scenario.
    tmp_path : pathlib.Path
        Owned archive destination and temporary file namespace.
    change : str
        Actual late payload change, seal byte/identity change, or output race.
    """
    root = mutable_bundle
    seal = (root / "bundle.json").read_bytes()
    icon = root / "share/icons/snn-studio.svg"
    original_icon = icon.read_bytes()
    manifest = verify_bundle(root)
    target = tmp_path / "delivery.tar.gz"
    process = subprocess.Popen(
        [sys.executable, "-B", "-m", "sc_neurocore.studio.desktop.archive", str(root), str(target)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 40
        while True:
            assert process.poll() is None, "The real exporter exited before the mid-write boundary."
            temporary = list(tmp_path.glob(".snn-studio-archive-*"))
            if temporary and temporary[0].stat().st_size > 100_000:
                break
            assert time.monotonic() < deadline, (
                "The exporter never began writing its complete runtime."
            )
            time.sleep(0.01)
        os.killpg(process.pid, signal.SIGSTOP)
        if change == "payload":
            icon.write_bytes(original_icon + b"\n<!-- changed while exporting -->\n")
        elif change == "manifest-bytes":
            (root / "bundle.json").write_bytes(seal + b"\n")
        elif change == "manifest-identity":
            create_bundle_manifest(
                root, version="3.16.0+changed", compatibility_key=manifest.compatibility_key
            )
        else:
            target.write_bytes(b"owner won the publication race")
        os.killpg(process.pid, signal.SIGCONT)
        stdout, stderr = process.communicate(timeout=180)
        assert process.returncode == 1, stdout + stderr
        assert stdout == "" and "Traceback" not in stderr
        if change == "destination":
            assert target.read_bytes() == b"owner won the publication race"
        else:
            assert "changed during archive creation" in stderr
            assert not target.exists()
        assert not list(tmp_path.glob(".snn-studio-archive-*"))
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.communicate(timeout=10)
        icon.write_bytes(original_icon)
        (root / "bundle.json").write_bytes(seal)
    assert verify_bundle(root) == manifest
