# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Trusted checkpoint loading helpers

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, Mapping

import torch


class CheckpointTrustError(ValueError):
    """Raised when a checkpoint is not present in the trusted digest set."""


def _checkpoint_digest(path: Path) -> tuple[bytes, str]:
    data = path.read_bytes()
    return data, hashlib.sha256(data).hexdigest()


def safe_load_checkpoint(
    path: str | Path,
    *,
    trusted_sha256: Mapping[str, str],
    map_location: str | torch.device = "cpu",
) -> Any:
    """Load a tensor/state-dict checkpoint only after SHA-256 verification.

    The trust map accepts either the file name or the resolved full path as key.
    PyTorch is always invoked with ``weights_only=True`` after digest validation.
    """
    checkpoint_path = Path(path).expanduser().resolve()
    data, digest = _checkpoint_digest(checkpoint_path)
    expected = trusted_sha256.get(checkpoint_path.name) or trusted_sha256.get(str(checkpoint_path))
    if expected is None:
        raise CheckpointTrustError(
            f"No trusted SHA-256 registered for checkpoint: {checkpoint_path.name}"
        )
    if digest.lower() != expected.lower():
        raise CheckpointTrustError(f"SHA-256 mismatch for checkpoint: {checkpoint_path.name}")
    return torch.load(io.BytesIO(data), map_location=map_location, weights_only=True)
