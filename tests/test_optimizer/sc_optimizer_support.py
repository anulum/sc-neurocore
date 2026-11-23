# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_sc_optimizer.py

"""Tests for the stochastic optimizer module."""

from __future__ import annotations

import unittest
from typing import Any

import pytest
from sc_neurocore.optimizer.sc_optimizer import (
    SCOptimizer,
    HardwareBudget,
    LayerProfile,
    LayerConfig,
)


def _fake_sa_result(n: int, *, feasible: bool = True, with_pareto: bool = True) -> dict[str, Any]:
    return {
        "feasible": feasible,
        "layer_luts": [100] * n,
        "layer_power": [1.0] * n,
        "layer_accuracy": [0.9] * n,
        "pareto_luts": [100, 200] if with_pareto else [],
        "pareto_power": [1.0, 2.0] if with_pareto else [],
        "pareto_score": [0.9, 0.95] if with_pareto else [],
    }


def _install_fake_rust(monkeypatch: pytest.MonkeyPatch, sa_result: dict[str, Any]) -> None:
    import sc_neurocore.optimizer.sc_optimizer as mod

    monkeypatch.setattr(mod, "_HAS_RUST", True)
    monkeypatch.setattr(mod, "py_opt_sa_search", lambda *a, **k: sa_result, raising=False)
    monkeypatch.setattr(
        mod,
        "py_opt_extract_pareto",
        lambda luts, power, score: {"luts": luts, "power": power, "score": score},
        raising=False,
    )


__all__ = [
    "unittest",
    "SCOptimizer",
    "HardwareBudget",
    "LayerProfile",
    "LayerConfig",
    "_fake_sa_result",
    "_install_fake_rust",
]
