# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Reproducible sealed desktop archive export

"""Export an existing complete desktop bundle without changing its declared provenance."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import os
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

from sc_neurocore.studio.desktop import DesktopStartRefused
from sc_neurocore.studio.desktop.bundle import verify_bundle


@dataclass(frozen=True, slots=True)
class BundleArchive:
    """Identity of a completely written delivery archive.

    Attributes
    ----------
    build_id : str
        Verified input bundle identity, including its existing data-family key.
    sha256 : str
        SHA-256 of the compressed archive bytes, not a publisher signature.
    size_bytes : int
        Complete compressed archive length in bytes.
    """

    build_id: str
    sha256: str
    size_bytes: int


class _HashingReader(io.BufferedReader):
    def __init__(self, path: Path) -> None:
        super().__init__(io.FileIO(path, "r"))
        self.digest = hashlib.sha256()

    def read(self, size: int | None = -1) -> bytes:
        data = super().read(size)
        self.digest.update(data)
        return data


def create_bundle_archive(root: Path, destination: Path) -> BundleArchive:
    """Publish a reproducible gzip/tar archive of an already sealed real bundle.

    Parameters
    ----------
    root : pathlib.Path
        Complete existing payload accepted by ``verify_bundle``.
    destination : pathlib.Path
        New archive path in an existing directory outside the input bundle.
        Existing files and symlinks are never replaced.

    Returns
    -------
    BundleArchive
        Exact compressed byte identity and the unchanged input build identity.

    Raises
    ------
    DesktopStartRefused
        Output placement is invalid, input changes while read, or sealing refuses.
    OSError
        Reading, writing or atomic no-replace publication fails. No partial
        destination is published; this operation owns only its temporary file.

    Notes
    -----
    Member order, timestamps, owner names/IDs, gzip filename and compression
    level are fixed. Byte reproducibility assumes the same Python/zlib toolchain.
    Ordinary modes, internal symlinks and all sealed bytes are retained. This
    exports accepted inputs; it does not assemble or qualify a scientific runtime.
    """
    root = root.resolve()
    destination = destination.parent.resolve() / destination.name
    if destination.is_relative_to(root):
        raise DesktopStartRefused("The archive must be outside its input bundle.")
    if destination.exists() or destination.is_symlink():
        raise DesktopStartRefused("The archive destination already exists.")
    manifest = verify_bundle(root)
    manifest_bytes = (root / "bundle.json").read_bytes()
    descriptor, name = tempfile.mkstemp(prefix=".snn-studio-archive-", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            with (
                gzip.GzipFile(
                    filename="", mode="wb", compresslevel=6, fileobj=output, mtime=0
                ) as compressed,
                tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive,
            ):
                seal = tarfile.TarInfo("bundle.json")
                seal.size, seal.mode = len(manifest_bytes), 0o644
                archive.addfile(seal, io.BytesIO(manifest_bytes))
                for entry in manifest.entries:
                    member = tarfile.TarInfo(entry.path)
                    member.mode = entry.mode
                    if entry.kind == "directory":
                        member.type = tarfile.DIRTYPE
                        archive.addfile(member)
                    elif entry.kind == "symlink":
                        member.type, member.linkname = tarfile.SYMTYPE, entry.value
                        archive.addfile(member)
                    else:
                        with _HashingReader(root / entry.path) as stream:
                            member.size = os.fstat(stream.fileno()).st_size
                            archive.addfile(member, stream)
                            if stream.digest.hexdigest() != entry.value:
                                raise DesktopStartRefused(
                                    "The desktop bundle changed during archive creation."
                                )
            output.flush()
            os.fsync(output.fileno())
        if verify_bundle(root) != manifest or (root / "bundle.json").read_bytes() != manifest_bytes:
            raise DesktopStartRefused("The desktop bundle changed during archive creation.")
        with temporary.open("rb") as stream:
            digest = hashlib.sha256()
            while chunk := stream.read(1_048_576):
                digest.update(chunk)
        result = BundleArchive(manifest.build_id, digest.hexdigest(), temporary.stat().st_size)
        os.link(temporary, destination)
        return result
    finally:
        temporary.unlink()


def main() -> int:
    """Export a complete sealed payload through the operator's module command.

    Returns
    -------
    int
        Zero after publication, one after an authored or filesystem refusal.
        Invalid arguments use argparse's normal exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    try:
        result = create_bundle_archive(args.bundle, args.archive)
    except DesktopStartRefused as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except OSError:
        print(
            "The desktop archive could not be written. Check the input and output directories.",
            file=sys.stderr,
        )
        return 1
    print(f"{result.sha256}  {args.archive}")
    print(f"Bundle {result.build_id}; {result.size_bytes} bytes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
