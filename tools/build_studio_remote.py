# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Producer-owned Studio pull-deploy bundle

"""Package a built Studio remote without publishing or deploying it.

Run ``python -m tools.build_studio_remote --output-dir <new-directory>`` after
``npm run build`` in ``studio/frontend``. The existing federation verifier
checks a frozen copy of the input bytes before they enter the archive. The
archive contains the split capability envelope at its root; the adjacent
``studio-deploy.json`` pins its SHA-256 for the platform pull consumer.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_FRONTEND = _REPO / "studio" / "frontend"
_BUNDLE = "sc-neurocore-studio-remote.tar.gz"
_MAX_BYTES = 256 * 1024 * 1024


class StudioBundleRefusal(ValueError):
    """A bundle input refusal whose wording is deliberately safe for the caller."""


def _frontend_files(directory: Path) -> dict[str, bytes]:
    if directory.is_symlink() or not directory.is_dir():
        raise StudioBundleRefusal("Frontend input must be a real build directory.")
    files: dict[str, bytes] = {}
    total = 0
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise StudioBundleRefusal("Frontend build must not contain symbolic links.")
        if path.is_dir():
            continue
        if not path.is_file():
            raise StudioBundleRefusal(
                "Frontend build must contain only regular files and directories."
            )
        name = path.relative_to(directory).as_posix()
        if name in {"manifest.json", "studio-deploy.json", _BUNDLE}:
            raise StudioBundleRefusal("Frontend build contains a reserved deployment filename.")
        total += path.stat().st_size
        if total > _MAX_BYTES:
            raise StudioBundleRefusal("Frontend build exceeds the 256 MiB bundle input limit.")
        files[name] = path.read_bytes()
    if not files.get("remoteEntry.js") or not files.get("index.html"):
        raise StudioBundleRefusal("Frontend build needs non-empty remoteEntry.js and index.html.")
    return files


def _verify_frontend(directory: Path) -> None:
    try:
        result = subprocess.run(
            ["node", str(_FRONTEND / "scripts" / "verify-federation-build.mjs"), str(directory)],
            capture_output=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise StudioBundleRefusal(
            "Node.js federation verification could not be completed."
        ) from exc
    if result.returncode:
        raise StudioBundleRefusal("Frontend federation verification failed; run npm run build.")


def _write_archive(path: Path, files: dict[str, bytes]) -> None:
    with (
        path.open("xb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive,
    ):
        for name, data in sorted(files.items()):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            member.mode = 0o644
            member.mtime = 0
            archive.addfile(member, io.BytesIO(data))


def build_bundle(frontend_dist: Path, output_dir: Path) -> Path:
    """Write verified UI bytes and their producer manifest into a new release directory.

    Parameters
    ----------
    frontend_dist
        Built UI directory, normally ``studio/frontend/dist``. Links, special
        files, reserved deployment names and inputs over 256 MiB are refused.
    output_dir
        New destination for the archive and deploy descriptor. Existing output
        is never replaced. Temporary snapshots stay beside this destination.

    Returns
    -------
    pathlib.Path
        The completed ``studio-deploy.json``. Its presence follows the complete
        archive write; this function performs no upload or host mutation.

    Raises
    ------
    StudioBundleRefusal
        If the frontend is invalid, federation verification fails or the
        destination already exists.
    OSError
        If the filesystem cannot read inputs or finish the output.
    """
    try:
        from sc_neurocore.federation.manifest import build_manifest
        from tools.emit_studio_manifest import render
    except ImportError as exc:
        raise StudioBundleRefusal(
            "Federation SDK could not be loaded; use Python >=3.11 with the federation extra."
        ) from exc

    if output_dir.exists() or output_dir.is_symlink():
        raise StudioBundleRefusal(
            "Output directory already exists; choose a new release directory."
        )
    files = _frontend_files(frontend_dist)
    manifest = build_manifest()
    manifest_bytes = render().encode("utf-8")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".studio-remote-", dir=output_dir.parent) as temporary:
        stage = Path(temporary)
        snapshot = stage / "frontend"
        snapshot.mkdir()
        for name, data in files.items():
            path = snapshot / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        _verify_frontend(snapshot)
        files["manifest.json"] = manifest_bytes
        archive_path = stage / _BUNDLE
        _write_archive(archive_path, files)
        archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        descriptor = {
            "schema": "studio.deploy-bundle.v1",
            "studio": manifest.studio,
            "studio_version": manifest.studio_version,
            "bundle_asset": _BUNDLE,
            "bundle_sha256": archive_sha256,
            "backend_compatibility": {
                "distribution": "sc-neurocore",
                "version": manifest.studio_version,
                "contract_era": manifest.contract_era,
                "platform_sdk": manifest.platform_sdk,
                "protocol_version": manifest.protocol_version,
                "content_digest": manifest.content_digest,
                "acceptance": "requires-matching-installed-backend-and-host-receipts",
            },
            "files": {
                name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                for name, data in sorted(files.items())
            },
        }
        output_dir.mkdir()
        with archive_path.open("rb") as source, (output_dir / _BUNDLE).open("xb") as destination:
            shutil.copyfileobj(source, destination)
        descriptor_path = output_dir / "studio-deploy.json"
        with descriptor_path.open("x", encoding="utf-8") as destination:
            destination.write(json.dumps(descriptor, indent=2, sort_keys=True) + "\n")
    return descriptor_path


def main(argv: list[str] | None = None) -> int:
    """Build a local producer bundle from the CLI without publication effects.

    Parameters
    ----------
    argv
        Explicit arguments or ``None`` for the process arguments.

    Returns
    -------
    int
        Zero on success, two for a deliberately refused input, or one for an
        unexpected filesystem failure. Diagnostics never quote exception text.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--frontend-dist", type=Path, default=_FRONTEND / "dist")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        descriptor = build_bundle(args.frontend_dist, args.output_dir)
    except StudioBundleRefusal as exc:
        print(str(exc))
        return 2
    except OSError:
        print("Studio remote bundle could not be written.")
        return 1
    print(descriptor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
