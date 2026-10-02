# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual candidate state divergence provenance

"""Distinguish real equation state guards from NumPy floating-point errors."""

from __future__ import annotations

import numpy as np

from sc_neurocore.studio.candidate_run import simulate_candidate
from tests.studio_candidate_support import adex_candidate


def test_candidate_divergence_preserves_the_authored_state_guard() -> None:
    """Report the equation's deliberate non-finite-state failure at its step."""
    document = adex_candidate()
    document["model"]["dynamics"]["v"] = "exp(v + 1000)"
    with np.errstate(over="ignore"):
        result = simulate_candidate(document, current=0.0, steps=10)
    assert result["diverged_at_step"] == 0
    assert result["final_state"] is None
    assert result["divergence"] == "'v' became non-finite (inf) after a euler step"


def test_candidate_divergence_does_not_publish_numpy_exception_text() -> None:
    """Map an actual NumPy overflow to the fixed candidate divergence reason."""
    document = adex_candidate()
    document["model"]["dynamics"]["v"] = "exp(v + 1000)"
    with np.errstate(over="raise"):
        result = simulate_candidate(document, current=0.0, steps=10)
    assert result["diverged_at_step"] == 0
    assert result["final_state"] is None
    assert result["divergence"] == "the candidate state could not remain finite"


def test_candidate_divergence_retains_the_authored_sqrt_domain_reason() -> None:
    """Report the existing square-root domain refusal at the failed step."""
    document = adex_candidate()
    document["model"]["dynamics"]["v"] = "sqrt(v)"
    result = simulate_candidate(document, current=0.0, steps=10)
    assert result["diverged_at_step"] == 0
    assert result["final_state"] is None
    assert result["trace"] == {"v": [], "w": []}
    assert result["divergence"] == "sqrt domain error"
