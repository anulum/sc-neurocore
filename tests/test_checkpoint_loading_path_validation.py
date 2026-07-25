# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Trusted checkpoint path and size validation

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from sc_neurocore.security.checkpoint_loading import CheckpointTrustError, safe_load_checkpoint


def test_safe_load_checkpoint_rejects_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pt"
    with pytest.raises(CheckpointTrustError, match="does not exist"):
        safe_load_checkpoint(missing, trusted_sha256={missing.name: "0" * 64})


def test_safe_load_checkpoint_rejects_symlink_path(tmp_path: Path) -> None:
    checkpoint = tmp_path / "weights.pt"
    torch.save({"weight": torch.tensor([1.0])}, checkpoint)
    symlink = tmp_path / "weights_link.pt"
    symlink.symlink_to(checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with pytest.raises(CheckpointTrustError, match="must not be a symlink"):
        safe_load_checkpoint(symlink, trusted_sha256={symlink.name: digest})


def test_safe_load_checkpoint_rejects_directory_path(tmp_path: Path) -> None:
    directory = tmp_path / "weights_dir.pt"
    directory.mkdir()
    with pytest.raises(CheckpointTrustError, match="not a regular file"):
        safe_load_checkpoint(directory, trusted_sha256={directory.name: "0" * 64})


def test_safe_load_checkpoint_rejects_oversized_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "weights.pt"
    torch.save({"weight": torch.tensor([1.0])}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with pytest.raises(CheckpointTrustError, match="exceeds maximum allowed size"):
        safe_load_checkpoint(checkpoint, trusted_sha256={checkpoint.name: digest}, max_bytes=1)


def test_safe_load_checkpoint_rejects_invalid_max_bytes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "weights.pt"
    torch.save({"weight": torch.tensor([1.0])}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with pytest.raises(CheckpointTrustError, match="max_bytes must be a positive integer"):
        safe_load_checkpoint(checkpoint, trusted_sha256={checkpoint.name: digest}, max_bytes=0)
