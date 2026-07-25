# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Trusted checkpoint digest contracts

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from sc_neurocore.security.checkpoint_loading import CheckpointTrustError, safe_load_checkpoint


def test_safe_load_checkpoint_rejects_unregistered_digest(tmp_path: Path) -> None:
    checkpoint = tmp_path / "weights.pt"
    torch.save({"weight": torch.tensor([1.0])}, checkpoint)

    with pytest.raises(CheckpointTrustError, match="No trusted SHA-256"):
        safe_load_checkpoint(checkpoint, trusted_sha256={})


def test_safe_load_checkpoint_rejects_mismatched_digest(tmp_path: Path) -> None:
    checkpoint = tmp_path / "weights.pt"
    torch.save({"weight": torch.tensor([1.0])}, checkpoint)

    with pytest.raises(CheckpointTrustError, match="SHA-256 mismatch"):
        safe_load_checkpoint(checkpoint, trusted_sha256={checkpoint.name: "0" * 64})


@pytest.mark.parametrize("bad_digest", ["abc123", "g" * 64, " " * 64])
def test_safe_load_checkpoint_rejects_invalid_trusted_digest_format(
    tmp_path: Path, bad_digest: str
) -> None:
    checkpoint = tmp_path / "weights.pt"
    torch.save({"weight": torch.tensor([1.0])}, checkpoint)
    with pytest.raises(CheckpointTrustError, match="Invalid trusted SHA-256 digest format"):
        safe_load_checkpoint(checkpoint, trusted_sha256={checkpoint.name: bad_digest})


def test_safe_load_checkpoint_rejects_empty_trusted_key(tmp_path: Path) -> None:
    checkpoint = tmp_path / "weights.pt"
    torch.save({"weight": torch.tensor([1.0])}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with pytest.raises(CheckpointTrustError, match="keys must be non-empty strings"):
        safe_load_checkpoint(checkpoint, trusted_sha256={"": digest})


def test_safe_load_checkpoint_uses_weights_only_after_digest_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "weights.pt"
    torch.save({"weight": torch.tensor([1.0])}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    calls: list[dict[str, object]] = []

    def fake_torch_load(buffer: object, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        assert isinstance(buffer, io.BytesIO)
        return {"weight": torch.tensor([2.0])}

    monkeypatch.setattr(torch, "load", fake_torch_load)

    loaded = safe_load_checkpoint(
        checkpoint,
        trusted_sha256={checkpoint.name: digest},
        map_location="cpu",
    )

    assert loaded["weight"].item() == 2.0
    assert calls == [{"map_location": "cpu", "weights_only": True}]


def test_safe_load_checkpoint_accepts_consistent_name_and_path_digests(tmp_path: Path) -> None:
    checkpoint = tmp_path / "weights.pt"
    torch.save({"weight": torch.tensor([1.0])}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    loaded = safe_load_checkpoint(
        checkpoint,
        trusted_sha256={checkpoint.name: digest, str(checkpoint.resolve()): digest},
        map_location="cpu",
    )
    assert isinstance(loaded, dict)
    assert torch.equal(loaded["weight"], torch.tensor([1.0]))


def test_safe_load_checkpoint_rejects_conflicting_name_and_path_digests(tmp_path: Path) -> None:
    checkpoint = tmp_path / "weights.pt"
    torch.save({"weight": torch.tensor([1.0])}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with pytest.raises(CheckpointTrustError, match="Conflicting trusted SHA-256 digests"):
        safe_load_checkpoint(
            checkpoint,
            trusted_sha256={checkpoint.name: digest, str(checkpoint.resolve()): "0" * 64},
            map_location="cpu",
        )
