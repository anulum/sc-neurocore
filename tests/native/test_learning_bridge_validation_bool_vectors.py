# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Autonomous-learning boolean-vector validation tests

"""Binary semantics and rejection contracts for boolean vector inputs."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore._native import learning_validation as validation


def test_bool_vectors_preserve_binary_semantics() -> None:
    direct = validation.as_bool_vector(np.array([True, False]), name="flags", length=2)
    numeric = validation.as_bool_vector([0, 1], name="flags")
    assert direct.dtype == np.bool_ and direct.flags.c_contiguous
    assert numeric.tolist() == [False, True]


@pytest.mark.parametrize("values", [[0, 2], [0.0, np.nan]])
def test_bool_vectors_reject_non_binary_values(values: object) -> None:
    with pytest.raises(ValueError, match="boolean, 0, or 1"):
        validation.as_bool_vector(values, name="flags")


def test_bool_vectors_reject_non_numeric_values() -> None:
    with pytest.raises(TypeError, match="booleans or binary"):
        validation.as_bool_vector([object()], name="flags")
