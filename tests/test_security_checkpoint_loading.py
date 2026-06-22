# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Tests for trusted checkpoint deserialisation guards

from __future__ import annotations

import hashlib
import io

import pytest

torch = pytest.importorskip("torch")

from sc_neurocore.security.checkpoint_loading import (
    CheckpointTrustError,
    _checkpoint_digest,
    safe_load_legacy_checkpoint,
    safe_load_checkpoint,
)


def test_safe_load_checkpoint_rejects_unregistered_digest(tmp_path):
    checkpoint = tmp_path / "weights.pt"
    torch.save({"weight": torch.tensor([1.0])}, checkpoint)

    with pytest.raises(CheckpointTrustError, match="No trusted SHA-256"):
        safe_load_checkpoint(checkpoint, trusted_sha256={})


def test_safe_load_checkpoint_rejects_mismatched_digest(tmp_path):
    checkpoint = tmp_path / "weights.pt"
    torch.save({"weight": torch.tensor([1.0])}, checkpoint)

    with pytest.raises(CheckpointTrustError, match="SHA-256 mismatch"):
        safe_load_checkpoint(checkpoint, trusted_sha256={checkpoint.name: "0" * 64})


@pytest.mark.parametrize("bad_digest", ["abc123", "g" * 64, " " * 64])
def test_safe_load_checkpoint_rejects_invalid_trusted_digest_format(tmp_path, bad_digest):
    checkpoint = tmp_path / "weights.pt"
    torch.save({"weight": torch.tensor([1.0])}, checkpoint)
    with pytest.raises(CheckpointTrustError, match="Invalid trusted SHA-256 digest format"):
        safe_load_checkpoint(checkpoint, trusted_sha256={checkpoint.name: bad_digest})


def test_safe_load_checkpoint_rejects_empty_trusted_key(tmp_path):
    checkpoint = tmp_path / "weights.pt"
    torch.save({"weight": torch.tensor([1.0])}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with pytest.raises(CheckpointTrustError, match="keys must be non-empty strings"):
        safe_load_checkpoint(checkpoint, trusted_sha256={"": digest})


def test_safe_load_checkpoint_rejects_missing_path(tmp_path):
    missing = tmp_path / "missing.pt"
    with pytest.raises(CheckpointTrustError, match="does not exist"):
        safe_load_checkpoint(missing, trusted_sha256={missing.name: "0" * 64})


def test_safe_load_checkpoint_rejects_symlink_path(tmp_path):
    checkpoint = tmp_path / "weights.pt"
    torch.save({"weight": torch.tensor([1.0])}, checkpoint)
    symlink = tmp_path / "weights_link.pt"
    symlink.symlink_to(checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with pytest.raises(CheckpointTrustError, match="must not be a symlink"):
        safe_load_checkpoint(symlink, trusted_sha256={symlink.name: digest})


def test_safe_load_checkpoint_rejects_directory_path(tmp_path):
    directory = tmp_path / "weights_dir.pt"
    directory.mkdir()
    with pytest.raises(CheckpointTrustError, match="not a regular file"):
        safe_load_checkpoint(directory, trusted_sha256={directory.name: "0" * 64})


def test_safe_load_checkpoint_rejects_oversized_file(tmp_path):
    checkpoint = tmp_path / "weights.pt"
    torch.save({"weight": torch.tensor([1.0])}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with pytest.raises(CheckpointTrustError, match="exceeds maximum allowed size"):
        safe_load_checkpoint(checkpoint, trusted_sha256={checkpoint.name: digest}, max_bytes=1)


def test_safe_load_checkpoint_rejects_invalid_max_bytes(tmp_path):
    checkpoint = tmp_path / "weights.pt"
    torch.save({"weight": torch.tensor([1.0])}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with pytest.raises(CheckpointTrustError, match="max_bytes must be a positive integer"):
        safe_load_checkpoint(checkpoint, trusted_sha256={checkpoint.name: digest}, max_bytes=0)


def test_safe_load_checkpoint_uses_weights_only_after_digest_match(tmp_path, monkeypatch):
    checkpoint = tmp_path / "weights.pt"
    torch.save({"weight": torch.tensor([1.0])}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    calls: list[dict[str, object]] = []

    def fake_torch_load(buffer, **kwargs):
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


def test_safe_load_checkpoint_accepts_consistent_name_and_path_digests(tmp_path):
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


def test_safe_load_checkpoint_rejects_conflicting_name_and_path_digests(tmp_path):
    checkpoint = tmp_path / "weights.pt"
    torch.save({"weight": torch.tensor([1.0])}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with pytest.raises(CheckpointTrustError, match="Conflicting trusted SHA-256 digests"):
        safe_load_checkpoint(
            checkpoint,
            trusted_sha256={checkpoint.name: digest, str(checkpoint.resolve()): "0" * 64},
            map_location="cpu",
        )


def test_safe_load_legacy_checkpoint_requires_explicit_digest(tmp_path):
    checkpoint = tmp_path / "metadata.pth"
    torch.save({"net": {"weight": torch.tensor([1.0])}, "acc": 75.0}, checkpoint)

    with pytest.raises(CheckpointTrustError, match="No trusted SHA-256"):
        safe_load_legacy_checkpoint(checkpoint, trusted_sha256={})


def test_safe_load_legacy_checkpoint_uses_weights_only_false_after_digest_match(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "metadata.pth"
    torch.save({"net": {"weight": torch.tensor([1.0])}, "acc": 75.0}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    calls: list[dict[str, object]] = []

    def fake_torch_load(buffer, **kwargs):
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


def test_safe_load_legacy_checkpoint_rejects_conflicting_name_and_path_digests(tmp_path):
    checkpoint = tmp_path / "metadata.pth"
    torch.save({"net": {"weight": torch.tensor([1.0])}, "acc": 75.0}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with pytest.raises(CheckpointTrustError, match="Conflicting trusted SHA-256 digests"):
        safe_load_legacy_checkpoint(
            checkpoint,
            trusted_sha256={checkpoint.name: digest, str(checkpoint.resolve()): "0" * 64},
            map_location="cpu",
        )


def test_safe_load_legacy_checkpoint_rejects_invalid_trusted_digest_format(tmp_path):
    checkpoint = tmp_path / "metadata.pth"
    torch.save({"net": {"weight": torch.tensor([1.0])}, "acc": 75.0}, checkpoint)
    with pytest.raises(CheckpointTrustError, match="Invalid trusted SHA-256 digest format"):
        safe_load_legacy_checkpoint(checkpoint, trusted_sha256={checkpoint.name: "bad"})


def test_safe_load_legacy_checkpoint_rejects_symlink_path(tmp_path):
    checkpoint = tmp_path / "metadata.pth"
    torch.save({"net": {"weight": torch.tensor([1.0])}, "acc": 75.0}, checkpoint)
    symlink = tmp_path / "metadata_link.pth"
    symlink.symlink_to(checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with pytest.raises(CheckpointTrustError, match="must not be a symlink"):
        safe_load_legacy_checkpoint(symlink, trusted_sha256={symlink.name: digest})


def test_safe_load_legacy_checkpoint_rejects_oversized_file(tmp_path):
    checkpoint = tmp_path / "metadata.pth"
    torch.save({"net": {"weight": torch.tensor([1.0])}, "acc": 75.0}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with pytest.raises(CheckpointTrustError, match="exceeds maximum allowed size"):
        safe_load_legacy_checkpoint(
            checkpoint,
            trusted_sha256={checkpoint.name: digest},
            max_bytes=1,
        )


def test_safe_load_legacy_checkpoint_rejects_non_positive_max_bytes(tmp_path):
    checkpoint = tmp_path / "metadata.pth"
    with pytest.raises(CheckpointTrustError, match="max_bytes must be a positive integer"):
        safe_load_legacy_checkpoint(checkpoint, trusted_sha256={}, max_bytes=0)


def test_safe_load_legacy_checkpoint_rejects_missing_file(tmp_path):
    missing = tmp_path / "absent.pth"
    with pytest.raises(CheckpointTrustError, match="does not exist"):
        safe_load_legacy_checkpoint(missing, trusted_sha256={missing.name: "0" * 64})


def test_safe_load_legacy_checkpoint_rejects_digest_mismatch(tmp_path):
    checkpoint = tmp_path / "metadata.pth"
    torch.save({"net": {"weight": torch.tensor([1.0])}, "acc": 75.0}, checkpoint)
    with pytest.raises(CheckpointTrustError, match="SHA-256 mismatch"):
        safe_load_legacy_checkpoint(checkpoint, trusted_sha256={checkpoint.name: "0" * 64})


def test_checkpoint_digest_rejects_missing_path(tmp_path):
    with pytest.raises(CheckpointTrustError, match="does not exist"):
        _checkpoint_digest(tmp_path / "absent.pth")


def test_checkpoint_digest_rejects_symlink_path(tmp_path):
    real = tmp_path / "real.pth"
    real.write_bytes(b"checkpoint bytes")
    link = tmp_path / "link.pth"
    link.symlink_to(real)
    with pytest.raises(CheckpointTrustError, match="must not be a symlink"):
        _checkpoint_digest(link)
