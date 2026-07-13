# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Torch learning precision controls

"""Validation and quantisation helpers for mixed-precision Torch learning."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from .learning_validation import require_integral, require_positive_float

MIN_PRECISION_BITS = 2
MAX_PRECISION_BITS = 31


def normalise_bit_spec(
    spec: int | Sequence[int] | np.ndarray[Any, Any] | torch.Tensor | None,
    *,
    count: int,
    device: torch.device,
    field: str,
) -> torch.Tensor | None:
    """Return a per-synapse integer bit vector within the supported domain."""
    if spec is None:
        return None
    if isinstance(spec, torch.Tensor):
        raw = spec.detach().to(device=device).flatten()
        if raw.dtype == torch.bool:
            raise TypeError(f"{field} entries must be integers, not bool")
        if raw.is_floating_point():
            if not bool(torch.all(torch.isfinite(raw)).item()):
                raise ValueError(f"{field} entries must be finite integers")
            if not bool(torch.all(raw == torch.round(raw)).item()):
                raise TypeError(f"{field} entries must be integers")
        bits = raw.to(dtype=torch.int64)
    elif isinstance(spec, np.ndarray):
        array = np.asarray(spec).reshape(-1)
        if array.dtype == np.bool_:
            raise TypeError(f"{field} entries must be integers, not bool")
        if not np.issubdtype(array.dtype, np.integer):
            if not np.issubdtype(array.dtype, np.floating):
                raise TypeError(f"{field} entries must be integers")
            if not np.all(np.isfinite(array)) or not np.all(array == np.rint(array)):
                raise TypeError(f"{field} entries must be finite integers")
        bits = torch.as_tensor(array, dtype=torch.int64, device=device)
    elif isinstance(spec, Sequence) and not isinstance(spec, (str, bytes, bytearray)):
        values = [
            require_integral(name=f"{field}[{index}]", value=value)
            for index, value in enumerate(spec)
        ]
        bits = torch.tensor(values, dtype=torch.int64, device=device)
    else:
        bits = torch.tensor(
            [require_integral(name=field, value=spec)], dtype=torch.int64, device=device
        )
    if bits.numel() == 1:
        bits = bits.expand(count)
    if bits.numel() != count:
        raise ValueError(f"{field} must be scalar or have length {count}, got {bits.numel()}")
    if bool(torch.any(bits < MIN_PRECISION_BITS).item()) or bool(
        torch.any(bits > MAX_PRECISION_BITS).item()
    ):
        raise ValueError(
            f"{field} entries must be in {MIN_PRECISION_BITS}..={MAX_PRECISION_BITS} bits"
        )
    return bits


def normalise_clip(value: object, *, field: str) -> float:
    """Return a finite, positive symmetric quantisation limit."""
    return require_positive_float(name=field, value=value)


def quantise_tensor(
    values: torch.Tensor,
    bits: torch.Tensor | None,
    clip: float,
) -> torch.Tensor:
    """Symmetrically quantise a tensor, preserving its device and dtype."""
    if bits is None:
        return values
    levels = torch.pow(2.0, bits.to(device=values.device, dtype=values.dtype) - 1.0) - 1.0
    clipped = torch.clamp(values, min=-clip, max=clip)
    result: torch.Tensor = torch.round(clipped * (levels / clip)) * (clip / levels)
    return result
