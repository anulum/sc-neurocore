# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — conductance-rate reference contracts

"""Pin the exact transcendental semantics shared by conductance references."""

from __future__ import annotations

import numpy as np

from tests.cosim_reference_conductance_rates import _np_exp, _reference_exprel


def test_numpy_exponential_preserves_schema_runner_semantics() -> None:
    argument = 0.75
    assert _np_exp(argument) == float(np.exp(argument))


def test_exprel_zero_uses_the_removable_singularity_limit() -> None:
    assert _reference_exprel(0.0) == 1.0


def test_exprel_near_zero_uses_the_linear_limit_term() -> None:
    argument = 5e-10
    assert _reference_exprel(argument) == 1.0 + argument / 2.0


def test_exprel_regular_branch_matches_numpy_expm1() -> None:
    argument = -0.25
    assert _reference_exprel(argument) == float(np.expm1(argument)) / argument
