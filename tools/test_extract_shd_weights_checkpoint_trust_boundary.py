# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCheckpointTrustBoundary from former test_extract_shd_weights.py

"""Focused suite: TestCheckpointTrustBoundary from former test_extract_shd_weights.py."""

from __future__ import annotations

import pytest
from extract_shd_weights_support import (
    _sha256,
    extract,
    torch,
)
from pathlib import (
    Path,
)
from sc_neurocore.security.checkpoint_loading import (
    CheckpointTrustError,
)
from typing import (
    cast,
)


class TestCheckpointTrustBoundary:
    """Refuse checkpoints whose digest, structure or metadata cannot be trusted."""

    def test_extract_requires_matching_checkpoint_sha256(self, tmp_path: Path) -> None:
        """A checkpoint whose SHA-256 differs from the expected digest is refused."""
        checkpoint = tmp_path / "shd_metadata.pth"
        torch.save({"net": {}, "acc": 0.0, "sigma": 0.23, "epoch": 0}, checkpoint)

        with pytest.raises(CheckpointTrustError, match="SHA-256 mismatch"):
            extract(str(checkpoint), str(tmp_path / "artifacts"), checkpoint_sha256="0" * 64)

    def test_extract_rejects_missing_checkpoint_sha256(self, tmp_path: Path) -> None:
        """Extraction without an expected checkpoint digest is refused."""
        checkpoint = tmp_path / "shd_metadata.pth"
        torch.save({"net": {}, "acc": 0.0, "sigma": 0.23, "epoch": 0}, checkpoint)

        with pytest.raises(ValueError, match="checkpoint_sha256 is required"):
            extract(str(checkpoint), str(tmp_path / "artifacts"), checkpoint_sha256=cast(str, None))

    @pytest.mark.parametrize("bad_digest", ["abc123", "g" * 64])
    def test_extract_rejects_invalid_checkpoint_sha256_format(
        self, tmp_path: Path, bad_digest: str
    ) -> None:
        """An expected digest that is not 64 hexadecimal characters is refused."""
        checkpoint = tmp_path / "shd_metadata.pth"
        torch.save({"net": {}, "acc": 0.0, "sigma": 0.23, "epoch": 0}, checkpoint)

        with pytest.raises(ValueError, match="64 hexadecimal characters"):
            extract(str(checkpoint), str(tmp_path / "artifacts"), checkpoint_sha256=bad_digest)

    def test_extract_rejects_missing_net_state_dict(self, tmp_path: Path) -> None:
        """A checkpoint without its 'net' state dictionary is refused."""
        checkpoint = tmp_path / "missing_net.pth"
        torch.save({"acc": 0.0, "sigma": 0.23, "epoch": 0}, checkpoint)

        with pytest.raises(ValueError, match="must contain a dictionary 'net'"):
            extract(
                str(checkpoint),
                str(tmp_path / "artifacts"),
                checkpoint_sha256=_sha256(str(checkpoint)),
            )

    def test_extract_rejects_non_tensor_net_values(self, tmp_path: Path) -> None:
        """A state dictionary with non-tensor values is refused."""
        checkpoint = tmp_path / "bad_net_values.pth"
        torch.save({"net": {"layers.1.weight": [1, 2, 3]}, "acc": 0.0, "sigma": 0.23}, checkpoint)

        with pytest.raises(ValueError, match="values must be tensors"):
            extract(
                str(checkpoint),
                str(tmp_path / "artifacts"),
                checkpoint_sha256=_sha256(str(checkpoint)),
            )

    def test_extract_rejects_non_finite_metadata(self, tmp_path: Path) -> None:
        """A non-finite accuracy in the checkpoint metadata is refused."""
        net = {
            "layers.1.weight": torch.zeros(128, 140),
            "layers.6.weight": torch.zeros(128, 128),
            "layers.10.weight": torch.zeros(20, 128),
            "layers.0.P": torch.zeros(140),
            "layers.5.P": torch.zeros(128),
        }
        checkpoint = tmp_path / "bad_meta.pth"
        torch.save({"net": net, "acc": float("nan"), "sigma": 0.23, "epoch": 1}, checkpoint)

        with pytest.raises(ValueError, match="metadata 'acc'"):
            extract(
                str(checkpoint),
                str(tmp_path / "artifacts"),
                checkpoint_sha256=_sha256(str(checkpoint)),
            )

    def test_extract_rejects_delay_length_mismatch(self, tmp_path: Path) -> None:
        """A delay vector whose length differs from its layer's inputs is refused."""
        net = {
            "layers.1.weight": torch.zeros(128, 140),
            "layers.6.weight": torch.zeros(128, 128),
            "layers.10.weight": torch.zeros(20, 128),
            "layers.0.P": torch.zeros(139),
            "layers.5.P": torch.zeros(128),
        }
        checkpoint = tmp_path / "bad_delay_len.pth"
        torch.save({"net": net, "acc": 0.0, "sigma": 0.23, "epoch": 1}, checkpoint)

        with pytest.raises(ValueError, match="unexpected delay length"):
            extract(
                str(checkpoint),
                str(tmp_path / "artifacts"),
                checkpoint_sha256=_sha256(str(checkpoint)),
            )

    def test_extract_rejects_out_of_range_delays(self, tmp_path: Path) -> None:
        """Delays outside the supported range of -15 to 15 are refused."""
        net = {
            "layers.1.weight": torch.zeros(128, 140),
            "layers.6.weight": torch.zeros(128, 128),
            "layers.10.weight": torch.zeros(20, 128),
            "layers.0.P": torch.full((140,), 16.0),
            "layers.5.P": torch.zeros(128),
        }
        checkpoint = tmp_path / "bad_delay_range.pth"
        torch.save({"net": net, "acc": 0.0, "sigma": 0.23, "epoch": 1}, checkpoint)

        with pytest.raises(ValueError, match="must stay within \\[-15, 15\\]"):
            extract(
                str(checkpoint),
                str(tmp_path / "artifacts"),
                checkpoint_sha256=_sha256(str(checkpoint)),
            )
