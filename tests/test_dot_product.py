# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

import numpy as np
import pytest

from sc_neurocore.synapses.dot_product import BitstreamDotProduct
from sc_neurocore.synapses.sc_synapse import BitstreamSynapse


def _synapse(w: float, length: int = 4096, seed: int = 42) -> BitstreamSynapse:
    return BitstreamSynapse(w_min=0.0, w_max=1.0, length=length, w=w, seed=seed)


def test_dot_product_all_ones_matches_weight_sum_until_clamp():
    synapses = [_synapse(0.2, seed=1), _synapse(0.3, seed=2)]
    dot = BitstreamDotProduct(synapses=synapses)
    pre_matrix = np.ones((2, 4096), dtype=np.uint8)

    post_matrix, y_scalar = dot.apply(pre_matrix)

    assert post_matrix.shape == pre_matrix.shape
    assert y_scalar == pytest.approx(0.5, abs=0.04)


def test_dot_product_output_clamps_to_output_range():
    synapses = [_synapse(0.8, seed=1), _synapse(0.8, seed=2)]
    dot = BitstreamDotProduct(synapses=synapses)
    pre_matrix = np.ones((2, 4096), dtype=np.uint8)

    _, y_scalar = dot.apply(pre_matrix, y_min=-2.0, y_max=2.0)

    assert y_scalar == pytest.approx(2.0)


def test_dot_product_requires_synapses_with_common_bitstream_length():
    synapses = [_synapse(0.5, length=256), _synapse(0.5, length=512)]

    with pytest.raises(ValueError, match="length"):
        BitstreamDotProduct(synapses=synapses)


@pytest.mark.parametrize(
    "synapses",
    [
        (),
        [_synapse(0.5), object()],
    ],
)
def test_dot_product_rejects_invalid_synapse_collection(synapses):
    with pytest.raises(ValueError, match="synapses"):
        BitstreamDotProduct(synapses=synapses)


@pytest.mark.parametrize(
    "pre_matrix",
    [
        [[0, 1, 0, 1]],
        np.ones(4096, dtype=np.uint8),
        np.ones((1, 4096), dtype=np.uint8),
        np.array([[0, 1, 2, 1], [1, 0, 1, 0]], dtype=np.uint8),
    ],
)
def test_invalid_pre_matrix_fails_closed(pre_matrix):
    dot = BitstreamDotProduct(synapses=[_synapse(0.5, length=4), _synapse(0.5, length=4)])

    with pytest.raises(ValueError, match="pre_matrix"):
        dot.apply(pre_matrix)


@pytest.mark.parametrize(
    ("y_min", "y_max"),
    [
        (float("nan"), 1.0),
        (0.0, float("inf")),
        (1.0, 1.0),
        (2.0, 1.0),
    ],
)
def test_invalid_output_range_fails_closed(y_min, y_max):
    dot = BitstreamDotProduct(synapses=[_synapse(0.5, length=4)])
    pre_matrix = np.ones((1, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="y_min"):
        dot.apply(pre_matrix, y_min=y_min, y_max=y_max)


def test_dot_product_rejects_pre_matrix_length_mismatch():
    # The matrix passes the input-count check but each row is shorter than the
    # synapses' shared bitstream length, so the dot product cannot be evaluated.
    dot = BitstreamDotProduct(synapses=[_synapse(0.5, length=256)])
    pre_matrix = np.ones((1, 128), dtype=np.uint8)

    with pytest.raises(ValueError, match="bitstream length"):
        dot.apply(pre_matrix)
