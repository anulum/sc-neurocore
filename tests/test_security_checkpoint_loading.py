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
