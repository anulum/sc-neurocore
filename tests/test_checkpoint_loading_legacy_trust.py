# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Trusted legacy checkpoint digest contracts

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from sc_neurocore.security.checkpoint_loading import (
    CheckpointTrustError,
    safe_load_legacy_checkpoint,
)


def test_safe_load_legacy_checkpoint_requires_explicit_digest(tmp_path: Path) -> None:
    checkpoint = tmp_path / "metadata.pth"
    torch.save({"net": {"weight": torch.tensor([1.0])}, "acc": 75.0}, checkpoint)

    with pytest.raises(CheckpointTrustError, match="No trusted SHA-256"):
        safe_load_legacy_checkpoint(checkpoint, trusted_sha256={})


def test_safe_load_legacy_checkpoint_uses_weights_only_false_after_digest_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "metadata.pth"
    torch.save({"net": {"weight": torch.tensor([1.0])}, "acc": 75.0}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    calls: list[dict[str, object]] = []

    def fake_torch_load(buffer: object, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        assert isinstance(buffer, io.BytesIO)
        return {"net": {"weight": torch.tensor([3.0])}, "acc": 75.0}

    monkeypatch.setattr(torch, "load", fake_torch_load)

    loaded = safe_load_legacy_checkpoint(
        checkpoint,
        trusted_sha256={checkpoint.name: digest},
        map_location="cpu",
    )

    assert loaded["net"]["weight"].item() == 3.0
    assert calls == [{"map_location": "cpu", "weights_only": False}]


def test_safe_load_legacy_checkpoint_rejects_conflicting_name_and_path_digests(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "metadata.pth"
    torch.save({"net": {"weight": torch.tensor([1.0])}, "acc": 75.0}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with pytest.raises(CheckpointTrustError, match="Conflicting trusted SHA-256 digests"):
        safe_load_legacy_checkpoint(
            checkpoint,
            trusted_sha256={checkpoint.name: digest, str(checkpoint.resolve()): "0" * 64},
            map_location="cpu",
        )


def test_safe_load_legacy_checkpoint_rejects_invalid_trusted_digest_format(tmp_path: Path) -> None:
    checkpoint = tmp_path / "metadata.pth"
    torch.save({"net": {"weight": torch.tensor([1.0])}, "acc": 75.0}, checkpoint)
    with pytest.raises(CheckpointTrustError, match="Invalid trusted SHA-256 digest format"):
        safe_load_legacy_checkpoint(checkpoint, trusted_sha256={checkpoint.name: "bad"})


def test_safe_load_legacy_checkpoint_rejects_digest_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "metadata.pth"
    torch.save({"net": {"weight": torch.tensor([1.0])}, "acc": 75.0}, checkpoint)
    with pytest.raises(CheckpointTrustError, match="SHA-256 mismatch"):
        safe_load_legacy_checkpoint(checkpoint, trusted_sha256={checkpoint.name: "0" * 64})
