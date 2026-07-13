# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Torch learning validation support

"""Torch-specific option and public-input validation helpers."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import torch

from .learning_validation import (
    RULE_BCM,
    RULE_ELIGENT,
    RULE_REWARD_STDP,
    require_non_negative_float,
    require_positive_float,
)

PRECISION_NAMES = ("weight", "trace", "eligibility", "theta", "act_avg")
KNOWN_KWARGS = frozenset(
    {
        "param_a_minus",
        "tau",
        "tau_plus",
        "tau_minus",
        "tau_e",
        "mixed_precision_bits",
        *(f"{name}_bits" for name in PRECISION_NAMES),
        *(f"{name}_clip" for name in PRECISION_NAMES),
    }
)


def rule_parameters(
    rule_type: int,
    param_a: object,
    param_b: object,
    kwargs: Mapping[str, Any],
) -> list[float]:
    """Build the five validated scalar parameters shared by every rule."""
    a = require_non_negative_float(name="param_a", value=param_a)
    b = require_non_negative_float(name="param_b", value=param_b)
    a_plus = max(a, 0.01 if rule_type == RULE_ELIGENT else 0.0001)
    default_minus = (
        0.001 if rule_type == RULE_ELIGENT else 1.0 if rule_type == RULE_BCM else a_plus * 0.5
    )
    a_minus = require_non_negative_float(
        name="param_a_minus", value=kwargs.get("param_a_minus", default_minus)
    )
    default_tau1 = (
        b
        if rule_type not in (RULE_ELIGENT, RULE_BCM)
        else 1.0
        if rule_type == RULE_ELIGENT
        else max(b, 1.0)
    )
    default_tau2 = b if rule_type not in (RULE_ELIGENT, RULE_BCM) else 1.0
    common_tau = kwargs.get("tau")
    tau_plus = require_positive_float(
        name="tau_plus",
        value=kwargs.get("tau_plus", default_tau1 if common_tau is None else common_tau),
    )
    tau_minus = require_positive_float(
        name="tau_minus",
        value=kwargs.get("tau_minus", default_tau2 if common_tau is None else common_tau),
    )
    default_tau_e = b if rule_type in (RULE_ELIGENT, RULE_REWARD_STDP) else 1.0
    tau_e = require_positive_float(name="tau_e", value=kwargs.get("tau_e", default_tau_e))
    return [a_plus, a_minus, tau_plus, tau_minus, tau_e]


def validate_input(
    values: torch.Tensor,
    *,
    name: str,
    count: int,
    device: torch.device,
    dtype: torch.dtype,
    probability: bool,
) -> torch.Tensor:
    """Move a finite one-dimensional public input onto the layer device."""
    if values.ndim != 1 or values.numel() != count:
        raise ValueError(f"{name} must have shape ({count},), got {tuple(values.shape)}")
    result = values.to(device=device, dtype=dtype)
    if result.device.type == "cpu" and result.numel() <= 64:
        scalars: list[float] = result.detach().tolist()
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError(f"{name} must contain only finite values")
        if probability and any(value < 0.0 or value > 1.0 for value in scalars):
            raise ValueError(f"{name} must contain values in [0, 1]")
        return result
    if not bool(torch.all(torch.isfinite(result)).item()):
        raise ValueError(f"{name} must contain only finite values")
    if probability and (
        bool(torch.any(result < 0.0).item()) or bool(torch.any(result > 1.0).item())
    ):
        raise ValueError(f"{name} must contain values in [0, 1]")
    return result
