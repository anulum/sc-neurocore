# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Checkpoint digest path validation

from __future__ import annotations

from pathlib import Path

import pytest

from sc_neurocore.security.checkpoint_loading import CheckpointTrustError, _checkpoint_digest


def test_checkpoint_digest_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(CheckpointTrustError, match="does not exist"):
        _checkpoint_digest(tmp_path / "absent.pth")


def test_checkpoint_digest_rejects_symlink_path(tmp_path: Path) -> None:
    real = tmp_path / "real.pth"
    real.write_bytes(b"checkpoint bytes")
    link = tmp_path / "link.pth"
    link.symlink_to(real)
    with pytest.raises(CheckpointTrustError, match="must not be a symlink"):
        _checkpoint_digest(link)
