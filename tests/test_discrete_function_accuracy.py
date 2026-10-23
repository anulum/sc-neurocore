# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Discontinuous fixed-point function references
"""Check actual error measurements and honest unary/binary domain coverage."""

from __future__ import annotations

import pytest
from sc_neurocore.compiler.lut_accuracy import measure


@pytest.mark.parametrize(("width", "fraction"), [(2, 0), (8, 4), (8, 7), (16, 8), (32, 24)])
@pytest.mark.parametrize("function", ["round", "heaviside"])
def test_discrete_accuracy_is_exact_without_claiming_sampled_domains_exhaustive(
    function: str, width: int, fraction: int
) -> None:
    """The measured error is zero; binary cardinality and sample admission stay explicit."""
    result = measure(function, width, fraction, lowering="v2")
    assert result.max_error == result.declared_error_bound == 0.0
    assert result.exhaustive == (width <= (8 if function == "heaviside" else 16))
    if function == "heaviside":
        public = result.to_public_dict()
        assert public["domain_cardinality"] == 1 << (2 * width)
        assert public["input_arity"] == 2
        assert len(result.worst_inputs) == len(result.operand_ranges) == 2
        assert (
            result.codes_evaluated == (1 << (2 * width))
            if result.exhaustive
            else result.codes_evaluated < (1 << (2 * width))
        )
