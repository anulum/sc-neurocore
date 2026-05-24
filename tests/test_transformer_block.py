# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

import numpy as np
import pytest

from sc_neurocore.transformers.block import StochasticTransformerBlock


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"d_model": 0, "n_heads": 1}, "d_model"),
        ({"d_model": 4, "n_heads": 0}, "n_heads"),
        ({"d_model": 5, "n_heads": 2}, "divisible"),
        ({"d_model": 4, "n_heads": 2, "length": 0}, "length"),
    ],
)
def test_transformer_block_rejects_invalid_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        StochasticTransformerBlock(**kwargs)


@pytest.mark.parametrize(
    "x,match",
    [
        (np.zeros((1, 1, 4)), "one- or two-dimensional"),
        (np.zeros(3), "trailing dimension"),
        (np.array([0.0, 1.0, np.inf, 0.5]), "finite"),
    ],
)
def test_transformer_block_rejects_invalid_forward_contracts(x, match):
    block = StochasticTransformerBlock(d_model=4, n_heads=2, length=16)

    with pytest.raises(ValueError, match=match):
        block.forward(x)


def test_multi_head_attention_rejects_wrong_internal_shape():
    block = StochasticTransformerBlock(d_model=4, n_heads=2, length=16)

    with pytest.raises(ValueError, match="shape"):
        block._multi_head_attention(np.zeros((2, 3)))
