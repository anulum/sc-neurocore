# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Trusted legacy checkpoint path and size validation

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from sc_neurocore.security.checkpoint_loading import (
    CheckpointTrustError,
    safe_load_legacy_checkpoint,
)


def test_safe_load_legacy_checkpoint_rejects_symlink_path(tmp_path: Path) -> None:
    checkpoint = tmp_path / "metadata.pth"
    torch.save({"net": {"weight": torch.tensor([1.0])}, "acc": 75.0}, checkpoint)
    symlink = tmp_path / "metadata_link.pth"
    symlink.symlink_to(checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with pytest.raises(CheckpointTrustError, match="must not be a symlink"):
        safe_load_legacy_checkpoint(symlink, trusted_sha256={symlink.name: digest})


def test_safe_load_legacy_checkpoint_rejects_oversized_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "metadata.pth"
    torch.save({"net": {"weight": torch.tensor([1.0])}, "acc": 75.0}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with pytest.raises(CheckpointTrustError, match="exceeds maximum allowed size"):
        safe_load_legacy_checkpoint(
            checkpoint,
            trusted_sha256={checkpoint.name: digest},
            max_bytes=1,
        )


def test_safe_load_legacy_checkpoint_rejects_non_positive_max_bytes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "metadata.pth"
    with pytest.raises(CheckpointTrustError, match="max_bytes must be a positive integer"):
        safe_load_legacy_checkpoint(checkpoint, trusted_sha256={}, max_bytes=0)


def test_safe_load_legacy_checkpoint_rejects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "absent.pth"
    with pytest.raises(CheckpointTrustError, match="does not exist"):
        safe_load_legacy_checkpoint(missing, trusted_sha256={missing.name: "0" * 64})
