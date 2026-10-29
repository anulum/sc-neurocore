# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio remote archive and producer CLI contracts

"""Exercise the real archive writer, CLI and Node verifier on transport artefacts.

The small ESM container tests archive custody, not production panel rendering.
Production UI build and host/browser acceptance remain distinct checks.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from tools.build_studio_remote import StudioBundleRefusal, build_bundle, main


@pytest.fixture(autouse=True)
def federation_sdk() -> None:
    """Require the optional federation SDK, supported by the Python 3.11+ CI lanes."""
    pytest.importorskip(
        "scpn_studio_platform", reason="requires the federation extra (Python >=3.11)"
    )


@pytest.fixture
def frontend(tmp_path: Path) -> Path:
    """Write a real ESM container and binary payload for archive transport checks."""
    root = tmp_path / "frontend"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text('<script src="/studios/sc-neurocore/remoteEntry.js"></script>')
    (root / "remoteEntry.js").write_text(
        'import{get as g,init as i}from"./assets/container.js";export{g as get,i as init};'
    )
    (root / "assets/container.js").write_text(
        'const sc_neurocore={"./SnnStudioPanel":()=>({default:()=>null})};'
        "export async function get(name){return sc_neurocore[name];}"
        "export function init(scope){return scope;}"
    )
    (root / "assets/trace.bin").write_bytes(bytes(range(256)))
    return root


@pytest.mark.parametrize("local_host", [False, True])
def test_cli_bundle_has_exact_root_manifest_and_all_asset_digests(
    frontend: Path, tmp_path: Path, local_host: bool
) -> None:
    """The process CLI emits a platform-admissible manifest and byte-bound archive."""
    from scpn_studio_platform.manifest import validate_studio_manifest

    output = tmp_path / "release"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.build_studio_remote",
            "--frontend-dist",
            str(frontend),
            "--output-dir",
            str(output),
            *(["--local-host"] if local_host else []),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    descriptor = json.loads((output / "studio-deploy.json").read_bytes())
    archive_path = output / descriptor["bundle_asset"]
    assert descriptor["schema"] == "studio.deploy-bundle.v1"
    assert descriptor["bundle_sha256"] == hashlib.sha256(archive_path.read_bytes()).hexdigest()
    with tarfile.open(archive_path) as archive:
        assert archive.getnames() == sorted(descriptor["files"])
        for member in archive:
            stream = archive.extractfile(member)
            assert stream is not None
            payload = stream.read()
            assert member.isfile() and member.mode == 0o644
            assert member.mtime == member.uid == member.gid == 0
            assert descriptor["files"][member.name] == {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
            if member.name == "manifest.json":
                envelope = json.loads(payload)
                assert set(envelope) == {"schema_a", "architecture_map"}
                schema_a = envelope["schema_a"]
                assert validate_studio_manifest(schema_a, supported_eras={"v2"}).admitted
                assert not validate_studio_manifest(schema_a, supported_eras={"v1"}).admitted
                assert schema_a["ui_module"]["exposes"] == ["./SnnStudioPanel"]
                if local_host:
                    assert schema_a["ui_module"]["capabilities"] == [
                        "federation:config",
                        "custom:sc-neurocore:api",
                    ]
                    assert (
                        schema_a["ui_module"]["remote_entry"]
                        == "/studios/sc-neurocore/remoteEntry.js"
                    )
                    assert schema_a["ui_module"]["remote_entry_digest"] == (
                        "sha256:"
                        + hashlib.sha256((frontend / "remoteEntry.js").read_bytes()).hexdigest()
                    )
                else:
                    assert schema_a["ui_module"]["remote_entry"] == (
                        "https://www.anulum.org/studios/sc-neurocore/remoteEntry.js"
                    )
                    assert "remote_entry_digest" not in schema_a["ui_module"]
                    assert "capabilities" not in schema_a["ui_module"]
                assert descriptor["backend_compatibility"]["version"] == schema_a["studio_version"]
            else:
                assert payload == (frontend / member.name).read_bytes()


@pytest.mark.parametrize("local_host", [False, True])
def test_bundle_is_reproducible_across_locations_and_source_metadata(
    frontend: Path, tmp_path: Path, local_host: bool
) -> None:
    """Source mode, mtime and destination names cannot change the release bytes."""
    first = build_bundle(frontend, tmp_path / "first", local_host=local_host)
    for path in frontend.rglob("*"):
        os.utime(path, (1700000000, 1700000000))
        if path.is_file():
            path.chmod(0o600)
    second = build_bundle(frontend, tmp_path / "second", local_host=local_host)
    assert first.read_bytes() == second.read_bytes()
    name = json.loads(first.read_bytes())["bundle_asset"]
    assert (first.parent / name).read_bytes() == (second.parent / name).read_bytes()


def test_asset_change_changes_the_deploy_digest(frontend: Path, tmp_path: Path) -> None:
    """An unchanged capability declaration cannot hide changed UI bytes from deployment."""
    first = json.loads(build_bundle(frontend, tmp_path / "first").read_bytes())
    (frontend / "assets/trace.bin").write_bytes(b"changed trace payload")
    second = json.loads(build_bundle(frontend, tmp_path / "second").read_bytes())
    assert first["bundle_sha256"] != second["bundle_sha256"]
    assert first["files"]["manifest.json"] == second["files"]["manifest.json"]


@pytest.mark.parametrize(
    "filename", ["manifest.json", "studio-deploy.json", "sc-neurocore-studio-remote.tar.gz"]
)
def test_reserved_deployment_names_are_refused(
    frontend: Path, tmp_path: Path, filename: str
) -> None:
    """A stale manifest or descriptor cannot replace the producer's own metadata."""
    (frontend / filename).write_bytes(b"stale deployment metadata")
    with pytest.raises(StudioBundleRefusal, match="reserved"):
        build_bundle(frontend, tmp_path / "release")
    assert not (tmp_path / "release").exists()


@pytest.mark.parametrize("filename", ["remoteEntry.js", "index.html"])
@pytest.mark.parametrize("empty", [False, True])
def test_missing_or_empty_entries_fail_before_output(
    frontend: Path, tmp_path: Path, filename: str, empty: bool
) -> None:
    """A deploy directory is never created for an incomplete frontend build."""
    if empty:
        (frontend / filename).write_bytes(b"")
    else:
        (frontend / filename).unlink()
    assert main(["--frontend-dist", str(frontend), "--output-dir", str(tmp_path / "release")]) == 2
    assert not (tmp_path / "release").exists()


@pytest.mark.parametrize("kind", ["root", "directory", "file"])
def test_symlinks_are_refused_without_following_them(
    frontend: Path, tmp_path: Path, kind: str
) -> None:
    """A link to another tree is never incorporated into a producer archive."""
    source = frontend
    if kind == "root":
        source = tmp_path / "linked-root"
        source.symlink_to(frontend, target_is_directory=True)
    else:
        (frontend / "link").symlink_to(
            frontend / "assets" if kind == "directory" else frontend / "index.html",
            target_is_directory=kind == "directory",
        )
    with pytest.raises(StudioBundleRefusal):
        build_bundle(source, tmp_path / "release")
    assert not (tmp_path / "release").exists()


def test_special_files_are_refused_without_reading(frontend: Path, tmp_path: Path) -> None:
    """A FIFO cannot block the archive collector or enter the release."""
    os.mkfifo(frontend / "pipe")
    with pytest.raises(StudioBundleRefusal, match="regular files"):
        build_bundle(frontend, tmp_path / "release")


def test_sparse_oversized_input_is_refused_before_reading(frontend: Path, tmp_path: Path) -> None:
    """The size ceiling refuses an oversized sparse asset without allocating its payload."""
    with (frontend / "assets/oversized.bin").open("wb") as stream:
        stream.truncate(256 * 1024 * 1024 + 1)
    with pytest.raises(StudioBundleRefusal, match="256 MiB"):
        build_bundle(frontend, tmp_path / "release")


def test_missing_build_and_existing_release_are_preserved(frontend: Path, tmp_path: Path) -> None:
    """Neither an absent build nor a conflicting destination can overwrite retained output."""
    with pytest.raises(StudioBundleRefusal, match="real build directory"):
        build_bundle(tmp_path / "absent", tmp_path / "release")
    output = tmp_path / "release"
    output.mkdir()
    (output / "keep").write_bytes(b"previous release")
    with pytest.raises(StudioBundleRefusal, match="already exists"):
        build_bundle(frontend, output)
    assert (output / "keep").read_bytes() == b"previous release"
    link = tmp_path / "dangling-output"
    link.symlink_to(tmp_path / "not-created", target_is_directory=True)
    with pytest.raises(StudioBundleRefusal, match="already exists"):
        build_bundle(frontend, link)
    assert link.is_symlink()


def test_missing_import_is_rejected_by_real_node_verifier(frontend: Path, tmp_path: Path) -> None:
    """The deployed entry must resolve its own runtime, not just exist by filename."""
    (frontend / "assets/container.js").unlink()
    with pytest.raises(StudioBundleRefusal, match="federation verification failed"):
        build_bundle(frontend, tmp_path / "release")
    assert not (tmp_path / "release").exists()
    assert not list(tmp_path.glob(".studio-remote-*"))


def test_node_unavailable_cli_has_fixed_refusal(frontend: Path, tmp_path: Path) -> None:
    """An actual missing Node executable refuses the CLI without interpreter diagnostics."""
    env = {**os.environ, "PATH": str(tmp_path / "empty-path")}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.build_studio_remote",
            "--frontend-dist",
            str(frontend),
            "--output-dir",
            str(tmp_path / "release"),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 2
    assert result.stdout.strip() == "Node.js federation verification could not be completed."
    assert result.stderr == ""
    assert not (tmp_path / "release").exists()


def test_missing_sdk_cli_has_fixed_refusal(frontend: Path, tmp_path: Path) -> None:
    """A real interpreter without site packages refuses rather than printing an import traceback."""
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            'import runpy,sys;sys.path[:]=[p for p in sys.path if "site-packages" not in p];'
            'runpy.run_module("tools.build_studio_remote",run_name="__main__")',
            "--frontend-dist",
            str(frontend),
            "--output-dir",
            str(tmp_path / "release"),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 2
    assert result.stdout.strip() == (
        "Federation SDK could not be loaded; use Python >=3.11 with the federation extra."
    )
    assert result.stderr == ""
    assert not (tmp_path / "release").exists()


def test_cli_filesystem_failure_has_fixed_diagnostic(
    frontend: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A file in the destination parent path yields a fixed CLI failure without overwriting it."""
    parent = tmp_path / "parent"
    parent.write_bytes(b"retained file")
    assert main(["--frontend-dist", str(frontend), "--output-dir", str(parent / "release")]) == 1
    assert capsys.readouterr().out.strip() == "Studio remote bundle could not be written."
    assert parent.read_bytes() == b"retained file"


def test_local_entry_pin_changes_with_actual_entry_and_binds_the_capability_surface(
    frontend: Path, tmp_path: Path
) -> None:
    """Local entry pins and declared digests track the entry, not a host-authored overlay."""
    from sc_neurocore.federation.manifest import build_manifest
    from tools.emit_studio_manifest import render

    original = json.loads(render())["schema_a"]
    assert original == build_manifest().to_dict()
    first = json.loads(
        build_bundle(frontend, tmp_path / "local-first", local_host=True).read_bytes()
    )
    with tarfile.open(tmp_path / "local-first" / first["bundle_asset"]) as archive:
        stream = archive.extractfile("manifest.json")
        assert stream is not None
        initial = json.load(stream)["schema_a"]
    assert initial["content_digest"] != original["content_digest"]
    assert initial["content_digest"] == first["backend_compatibility"]["content_digest"]
    (frontend / "remoteEntry.js").write_text((frontend / "remoteEntry.js").read_text() + "\n")
    second = json.loads(
        build_bundle(frontend, tmp_path / "local-second", local_host=True).read_bytes()
    )
    with tarfile.open(tmp_path / "local-second" / second["bundle_asset"]) as archive:
        stream = archive.extractfile("manifest.json")
        assert stream is not None
        changed = json.load(stream)["schema_a"]
    assert (
        initial["ui_module"]["remote_entry_digest"] != changed["ui_module"]["remote_entry_digest"]
    )
    assert initial["content_digest"] != changed["content_digest"]
    assert initial["studio_version"] == changed["studio_version"]
    assert json.loads(render())["schema_a"] == original
