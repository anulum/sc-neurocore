# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_resilience.py

"""Shared deterministic evaluator for the fault resilience suites."""

from __future__ import annotations

from typing import Any
import numpy as np


def _eval_fn(weights: list[np.ndarray[Any, Any]]) -> float:
    """Deterministic test evaluator: accuracy ~ mean absolute weight."""
    return float(np.clip(np.mean([np.abs(w).mean() for w in weights]), 0, 1))


__all__ = ["_eval_fn"]
