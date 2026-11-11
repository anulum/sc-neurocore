# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Remaining branches of the director, the decoder and analysis variability

"""Branches of the files that the spike-train statistics promotion changed.

The existing suites never reach four branches of these files: a moderate
E-E density in the diagnosis, a problem name that ``correct`` does not know,
a priming context without E-E weights, and the analysis variability module
loaded with the Rust helper switched off.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest

from sc_neurocore.analysis.spike_stats import variability
from sc_neurocore.identity import DirectorController, IdentitySubstrate, StateDecoder

_HEALTHY = {"mean_rate": 10.0, "cv": 1.0, "fano": 1.0, "perm_entropy": 0.8, "n_steps": 200}


def _substrate() -> IdentitySubstrate:
    """Return a small substrate after 100 simulated steps."""
    substrate = IdentitySubstrate(n_cortical=30, n_inhibitory=10, n_memory=5, seed=42)
    substrate.run(duration=0.1, dt=0.001)
    return substrate


def test_moderate_connectivity_is_not_a_problem() -> None:
    """Half of the E-E weights non-zero is neither too dense nor too sparse."""
    substrate = _substrate()
    weights = substrate.proj_ee.data
    weights[:] = 0.0
    weights[: weights.size // 2] = 0.5
    director = DirectorController(substrate)
    with patch.object(director, "monitor", return_value=dict(_HEALTHY)):
        assert director.diagnose() == []


def test_an_unrecognised_problem_changes_no_weights() -> None:
    """A problem name without a correction leaves every projection as it was."""
    substrate = _substrate()
    director = DirectorController(substrate)
    before = {
        name: getattr(substrate, name).data.copy() for name in ("proj_ee", "proj_ie", "proj_ii")
    }
    with patch.object(director, "diagnose", return_value=["not_a_known_problem"]):
        director.correct()
    for name, data in before.items():
        np.testing.assert_array_equal(getattr(substrate, name).data, data)
    with (
        patch.object(director, "monitor", return_value=dict(_HEALTHY)),
        patch.object(director, "diagnose", return_value=[]),
    ):
        assert "Corrections applied: 1" in director.report()


def test_priming_context_without_ee_weights_omits_the_weight_line() -> None:
    """A substrate with no E-E weights gives a context with no weight statistics."""
    rng = np.random.default_rng(11)
    history = [(rng.random(6) < 0.3).astype(np.int8) for _ in range(60)]
    substrate = SimpleNamespace(
        spike_history=history,
        n_cortical=6,
        ee_weights=np.zeros(0),
        health_check=lambda: {"is_healthy": True},
    )
    context = StateDecoder(cast(IdentitySubstrate, substrate)).generate_priming_context()
    assert context.startswith("Substrate active: 60 steps.")
    assert "Health: OK." in context
    assert "E-E weights" not in context


def test_variability_leaves_the_rust_helper_unloaded_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With SC_NEUROCORE_NO_RUST set, the module loads without the helper and still works."""
    monkeypatch.setenv("SC_NEUROCORE_NO_RUST", "1")
    try:
        module = importlib.reload(variability)
        assert module._HAS_RUST is False
        assert module._ssc is None
        train = (np.random.default_rng(4).random(200) < 0.3).astype(np.int8)
        assert 0.0 < module.permutation_entropy(train) <= 1.0
    finally:
        monkeypatch.delenv("SC_NEUROCORE_NO_RUST")
        importlib.reload(variability)
