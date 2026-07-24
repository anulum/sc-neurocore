# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (exact_flow_rejects) from former test_model_resonate_and_fire.py

from __future__ import annotations

from tests.model_resonate_and_fire_support import *  # noqa: F403


def test_exact_flow_rejects_zero_denominator() -> None:
    with pytest.raises(ValueError, match="denominator"):
        ResonateAndFireNeuron._exact_linear_flow(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )


def test_exact_flow_rejects_exponential_overflow() -> None:
    with pytest.raises(FloatingPointError, match="decay"):
        ResonateAndFireNeuron._exact_linear_flow(
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1000.0,
        )


def test_exact_flow_rejects_nonfinite_equilibrium() -> None:
    with pytest.raises(FloatingPointError, match="equilibrium"):
        ResonateAndFireNeuron._exact_linear_flow(
            0.0,
            0.0,
            1.0e308,
            1.0e-154,
            1.0e-154,
            0.01,
        )


def test_exact_flow_rejects_nonfinite_post_rotation_candidate() -> None:
    with pytest.raises(FloatingPointError, match="candidate"):
        ResonateAndFireNeuron._exact_linear_flow(
            1.0e308,
            1.0e308,
            0.0,
            1.0,
            1.0,
            1.0,
        )
