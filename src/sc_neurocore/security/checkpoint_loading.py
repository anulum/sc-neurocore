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
import re
from pathlib import Path
from typing import Any, Mapping

import torch


class CheckpointTrustError(ValueError):
    """Raised when a checkpoint is not present in the trusted digest set."""


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
DEFAULT_MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024  # 512 MiB


def _validate_trusted_map(trusted_sha256: Mapping[str, str]) -> None:
    if not trusted_sha256:
        return
    for key, digest in trusted_sha256.items():
        if not isinstance(key, str) or not key.strip():
            raise CheckpointTrustError("Trusted checkpoint map keys must be non-empty strings")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise CheckpointTrustError(
                f"Invalid trusted SHA-256 digest format for checkpoint key: {key}"
            )


def _resolve_expected_digest(
    checkpoint_path: Path,
    trusted_sha256: Mapping[str, str],
) -> str:
    by_name = trusted_sha256.get(checkpoint_path.name)
    by_path = trusted_sha256.get(str(checkpoint_path))
    if by_name is None and by_path is None:
        raise CheckpointTrustError(
            f"No trusted SHA-256 registered for checkpoint: {checkpoint_path.name}"
        )
    if by_name is not None and by_path is not None and by_name.lower() != by_path.lower():
        raise CheckpointTrustError(
            "Conflicting trusted SHA-256 digests for checkpoint basename and absolute path: "
            f"{checkpoint_path.name}"
        )
    return by_name if by_name is not None else by_path  # type: ignore[return-value]


def _checkpoint_digest(path: Path) -> tuple[bytes, str]:
    if not path.exists():
        raise CheckpointTrustError(f"Checkpoint path does not exist: {path}")
    if path.is_symlink():
        raise CheckpointTrustError(f"Checkpoint path must not be a symlink: {path}")
    if not path.is_file():
        raise CheckpointTrustError(f"Checkpoint path is not a regular file: {path}")
    data = path.read_bytes()
    return data, hashlib.sha256(data).hexdigest()


def safe_load_checkpoint(
    path: str | Path,
    *,
    trusted_sha256: Mapping[str, str],
    map_location: str | torch.device = "cpu",
    max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES,
) -> Any:
    """Load a tensor/state-dict checkpoint only after SHA-256 verification.

    The trust map accepts either the file name or the resolved full path as key.
    PyTorch is always invoked with ``weights_only=True`` after digest validation.
    """
    if not isinstance(max_bytes, int) or max_bytes <= 0:
        raise CheckpointTrustError(f"max_bytes must be a positive integer, got {max_bytes!r}")
    _validate_trusted_map(trusted_sha256)
    raw_path = Path(path).expanduser()
    if raw_path.is_symlink():
        raise CheckpointTrustError(f"Checkpoint path must not be a symlink: {raw_path}")
    checkpoint_path = raw_path.resolve()
    if not checkpoint_path.exists():
        raise CheckpointTrustError(f"Checkpoint path does not exist: {checkpoint_path}")
    if checkpoint_path.stat().st_size > max_bytes:
        raise CheckpointTrustError(
            f"Checkpoint exceeds maximum allowed size ({max_bytes} bytes): {checkpoint_path.name}"
        )
    data, digest = _checkpoint_digest(checkpoint_path)
    expected = _resolve_expected_digest(checkpoint_path, trusted_sha256)
    if digest.lower() != expected.lower():
        raise CheckpointTrustError(f"SHA-256 mismatch for checkpoint: {checkpoint_path.name}")
    return torch.load(io.BytesIO(data), map_location=map_location, weights_only=True)


def safe_load_legacy_checkpoint(
    path: str | Path,
    *,
    trusted_sha256: Mapping[str, str],
    map_location: str | torch.device = "cpu",
    max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES,
) -> Any:
    """Load a metadata checkpoint through pickle only after SHA-256 verification.

    Use this only for legacy internal checkpoints whose dictionaries contain
    non-state-dict metadata and cannot yet be represented by ``weights_only``.
    """
    if not isinstance(max_bytes, int) or max_bytes <= 0:
        raise CheckpointTrustError(f"max_bytes must be a positive integer, got {max_bytes!r}")
    _validate_trusted_map(trusted_sha256)
    raw_path = Path(path).expanduser()
    if raw_path.is_symlink():
        raise CheckpointTrustError(f"Checkpoint path must not be a symlink: {raw_path}")
    checkpoint_path = raw_path.resolve()
    if not checkpoint_path.exists():
        raise CheckpointTrustError(f"Checkpoint path does not exist: {checkpoint_path}")
    if checkpoint_path.stat().st_size > max_bytes:
        raise CheckpointTrustError(
            f"Checkpoint exceeds maximum allowed size ({max_bytes} bytes): {checkpoint_path.name}"
        )
    data, digest = _checkpoint_digest(checkpoint_path)
    expected = _resolve_expected_digest(checkpoint_path, trusted_sha256)
    if digest.lower() != expected.lower():
        raise CheckpointTrustError(f"SHA-256 mismatch for checkpoint: {checkpoint_path.name}")
    # Legacy metadata checkpoints require pickle; SHA-256 is verified above.
    return torch.load(  # nosec B614
        io.BytesIO(data), map_location=map_location, weights_only=False
    )
