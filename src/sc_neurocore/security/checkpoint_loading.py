# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Trusted checkpoint loading helpers

"""SHA-256-gated checkpoint loading helpers for trusted artefacts.

This module separates the safe tensor/state-dict path, which uses
``torch.load(..., weights_only=True)``, from the legacy metadata path that still
requires pickle after an explicit digest match. Callers must provide the trusted
digest map instead of accepting arbitrary downloaded checkpoints.
"""

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


def safe_load_legacy_checkpoint(
    path: str | Path,
    *,
    trusted_sha256: Mapping[str, str],
    map_location: str | torch.device = "cpu",
) -> Any:
    """Load a metadata checkpoint through pickle only after SHA-256 verification.

    Use this only for legacy internal checkpoints whose dictionaries contain
    non-state-dict metadata and cannot yet be represented by ``weights_only``.
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
    # Legacy metadata checkpoints require pickle; SHA-256 is verified above.
    return torch.load(  # nosec B614
        io.BytesIO(data), map_location=map_location, weights_only=False
    )
