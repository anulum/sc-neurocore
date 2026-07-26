# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Autonomous-learning numeric-vector validation tests

"""Shape, length, float32, finiteness, and probability-domain contracts."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore._native import learning_validation as validation


def test_vector_shape_and_length_are_exact() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        validation.as_bool_vector([[True]], name="flags")
    with pytest.raises(ValueError, match="length 2"):
        validation.as_float_vector([1.0], name="values", length=2)


def test_float_vectors_are_finite_contiguous_float32() -> None:
    result = validation.as_float_vector(np.array([1, 2], dtype=np.int64), name="values")
    assert result.dtype == np.float32 and result.flags.c_contiguous
    with pytest.raises(TypeError, match="numeric"):
        validation.as_float_vector([object()], name="values")
    with pytest.raises(ValueError, match="finite"):
        validation.as_float_vector([np.inf], name="values")


def test_probability_vectors_enforce_closed_unit_interval() -> None:
    result = validation.as_probability_vector([0.0, 1.0], name="probabilities")
    assert result.tolist() == [0.0, 1.0]
    for values in ([-0.1], [1.1]):
        with pytest.raises(ValueError, match="probabilities"):
            validation.as_probability_vector(values, name="probabilities")
