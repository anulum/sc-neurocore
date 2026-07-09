# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# SC-NeuroCore — Tests for MNIST SC checkpoint validator

from __future__ import annotations

import hashlib
import struct

import pytest

torch = pytest.importorskip("torch")

from sc_neurocore.security.checkpoint_loading import CheckpointTrustError
from tools.validate_mnist_sc_checkpoint import (
    _state_dict_from_checkpoint,
    load_mnist_idx,
    validate_checkpoint,
)


def _write_idx_fixture(root, *, image_magic=2051, label_magic=2049) -> None:
    raw = root / "MNIST" / "raw"
    raw.mkdir(parents=True)
    image_payload = bytes([0, 255, 128, 64])
    (raw / "t10k-images-idx3-ubyte").write_bytes(
        struct.pack(">IIII", image_magic, 1, 2, 2) + image_payload
    )
    (raw / "t10k-labels-idx1-ubyte").write_bytes(struct.pack(">II", label_magic, 1) + bytes([7]))


def test_load_mnist_idx_reads_local_raw_files(tmp_path):
    _write_idx_fixture(tmp_path)
    images, labels = load_mnist_idx(tmp_path, 1)

    assert images.shape == (1, 2, 2)
    assert labels.tolist() == [7]
    assert images.dtype.kind == "f"


def test_load_mnist_idx_rejects_bad_magic(tmp_path):
    _write_idx_fixture(tmp_path, image_magic=123)

    with pytest.raises(ValueError, match="magic"):
        load_mnist_idx(tmp_path, 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"samples": 0},
        {"timesteps": 0},
        {"bitstream_length": 0},
        {"min_sc_accuracy": -0.1},
        {"min_agreement": 1.1},
        {"checkpoint_sha256": None},
        {"checkpoint_sha256": "abc123"},
    ],
)
def test_validate_checkpoint_rejects_invalid_numeric_contract(tmp_path, kwargs):
    params = {
        "checkpoint": tmp_path / "missing.pt",
        "data_dir": tmp_path,
        "samples": 1,
        "timesteps": 1,
        "bitstream_length": 64,
        "seed": 0,
        "min_sc_accuracy": 0.0,
        "min_agreement": 0.0,
        "checkpoint_sha256": "0" * 64,
    }
    params.update(kwargs)

    with pytest.raises(ValueError):
        validate_checkpoint(**params)


def test_state_dict_from_checkpoint_verifies_optional_sha256(tmp_path):
    checkpoint = tmp_path / "conv.pt"
    torch.save({"layer.weight": torch.tensor([1.0])}, checkpoint)

    with pytest.raises(CheckpointTrustError, match="SHA-256 mismatch"):
        _state_dict_from_checkpoint(checkpoint, trusted_sha256={checkpoint.name: "0" * 64})


def test_state_dict_from_checkpoint_rejects_empty_mapping(tmp_path):
    checkpoint = tmp_path / "empty.pt"
    torch.save({}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="state_dict"):
        _state_dict_from_checkpoint(checkpoint, trusted_sha256={checkpoint.name: digest})


def test_state_dict_from_checkpoint_rejects_empty_tensor(tmp_path):
    checkpoint = tmp_path / "empty_tensor.pt"
    torch.save({"layer.weight": torch.empty(0)}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="must be non-empty"):
        _state_dict_from_checkpoint(checkpoint, trusted_sha256={checkpoint.name: digest})


def test_state_dict_from_checkpoint_rejects_non_finite_tensor(tmp_path):
    checkpoint = tmp_path / "nan_tensor.pt"
    torch.save({"layer.weight": torch.tensor([1.0, float("nan")])}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="non-finite"):
        _state_dict_from_checkpoint(checkpoint, trusted_sha256={checkpoint.name: digest})
