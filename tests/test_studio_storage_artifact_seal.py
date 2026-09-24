# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — authority-side artefact sealing

"""Sealed artefacts are private, read-only and never replace other bytes."""

from __future__ import annotations

import hashlib
from pathlib import Path
import stat

import pytest

from sc_neurocore.studio.platform.storage_artifact_seal import SealedArtifactWriter

JOB = "sj_" + "5" * 16


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    authority = tmp_path / "authority"
    authority.mkdir(mode=0o700)
    return authority


def test_artefacts_are_sealed_read_only_in_private_directories(root: Path) -> None:
    """Nested paths are created private; files are read-only with exact bytes."""
    with SealedArtifactWriter(root, JOB) as writer:
        writer.seal("reports/summary.json", b"{}", sha256=_digest(b"{}"))
        writer.seal("weights.bin", b"", sha256=_digest(b""))
    sealed = root / JOB / "reports" / "summary.json"
    assert sealed.read_bytes() == b"{}"
    assert stat.S_IMODE(sealed.stat().st_mode) == 0o400
    for directory in (root / JOB, root / JOB / "reports"):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert (root / JOB / "weights.bin").read_bytes() == b""
    assert sorted(path.name for path in (root / JOB).iterdir()) == ["reports", "weights.bin"]


def test_identical_retry_is_accepted_and_different_bytes_are_refused(root: Path) -> None:
    """A retry after a crash completes; other bytes never replace a sealed file."""
    with SealedArtifactWriter(root, JOB) as writer:
        writer.seal("result.bin", b"abc", sha256=_digest(b"abc"))
    # A crash after writing the partial copy leaves it behind, read-only.
    leftover = root / JOB / ".result.bin.partial"
    leftover.write_bytes(b"stale")
    leftover.chmod(0o400)
    with SealedArtifactWriter(root, JOB) as writer:
        writer.seal("result.bin", b"abc", sha256=_digest(b"abc"))
        with pytest.raises(FileExistsError, match="different artefact"):
            writer.seal("result.bin", b"xyz", sha256=_digest(b"xyz"))
    assert (root / JOB / "result.bin").read_bytes() == b"abc"
    assert not leftover.exists()


@pytest.mark.parametrize("occupant", ["symlink", "directory"])
def test_non_file_occupants_are_never_followed_or_replaced(
    root: Path, tmp_path: Path, occupant: str
) -> None:
    """A symbolic link or directory at the final path refuses the seal."""
    target = tmp_path / "outside.bin"
    target.write_bytes(b"abc")
    (root / JOB).mkdir(mode=0o700)
    final = root / JOB / "result.bin"
    if occupant == "symlink":
        final.symlink_to(target)
    else:
        final.mkdir()
    with SealedArtifactWriter(root, JOB) as writer, pytest.raises(FileExistsError):
        writer.seal("result.bin", b"abc", sha256=_digest(b"abc"))
    assert target.read_bytes() == b"abc"
    assert final.is_symlink() if occupant == "symlink" else final.is_dir()


@pytest.mark.parametrize("shared", ["root", "job", "nested"])
def test_directories_open_to_others_are_refused(root: Path, shared: str) -> None:
    """The root, the job directory and nested directories must stay private."""
    if shared == "root":
        root.chmod(0o755)
        with pytest.raises(PermissionError, match="root is not private"):
            SealedArtifactWriter(root, JOB)
        return
    (root / JOB).mkdir(mode=0o700)
    if shared == "job":
        (root / JOB).chmod(0o750)
        with pytest.raises(PermissionError, match="not private"):
            SealedArtifactWriter(root, JOB)
        return
    (root / JOB / "reports").mkdir(mode=0o755)
    (root / JOB / "reports").chmod(0o755)
    with (
        SealedArtifactWriter(root, JOB) as writer,
        pytest.raises(PermissionError, match="not private"),
    ):
        writer.seal("reports/summary.json", b"{}", sha256=_digest(b"{}"))
