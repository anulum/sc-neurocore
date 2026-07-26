# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Vector bitstream popcount tests

"""Known-pattern and zero-input packed popcount contracts."""

import numpy as np

from sc_neurocore.accel.vector_ops import pack_bitstream, vec_popcount


def test_vec_popcount_known() -> None:
    """vec_popcount should count total set bits."""
    bits = np.array([1, 0, 1, 1, 0, 1], dtype=np.uint8)
    packed = pack_bitstream(bits)
    count = vec_popcount(packed)
    assert count == 4


def test_vec_popcount_zero() -> None:
    """Popcount of all-zero input should be zero."""
    bits = np.zeros(128, dtype=np.uint8)
    packed = pack_bitstream(bits)
    assert vec_popcount(packed) == 0
