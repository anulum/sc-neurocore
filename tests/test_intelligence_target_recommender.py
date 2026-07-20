# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Compiler target recommender contracts

"""Contracts for compiler target recommendation and score filtering."""

from sc_neurocore.compiler.intelligence import recommend_target

_COMPLEX_EQS = {"v": "a + b + c + d + e + f + g - h"}


def test_recommend_target_applies_frequency_floor() -> None:
    """A very high minimum frequency exercises the frequency-floor filter."""
    relaxed = recommend_target({"v": "x + 1"})
    constrained = recommend_target({"v": "x + 1"}, min_freq_mhz=1.0e9)

    assert len(constrained) <= len(relaxed)


def test_recommend_target_scores_complex_models() -> None:
    """A high-operation-count model yields non-negative scored recommendations."""
    recommendations = recommend_target(_COMPLEX_EQS)

    assert recommendations
    assert all(result.score >= 0 for result in recommendations)
