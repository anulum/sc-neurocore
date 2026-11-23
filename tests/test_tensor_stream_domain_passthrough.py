# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDomainPassthrough from former test_tensor_stream.py

"""Focused suite: TestDomainPassthrough from former test_tensor_stream.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.core.tensor_stream import (
    TensorStream,
)


class TestDomainPassthrough:
    """Conversions into the same domain."""

    def test_prob_to_prob(self) -> None:
        """A probability tensor converts to itself."""
        ts = TensorStream.from_prob(np.array([0.42]))
        np.testing.assert_allclose(ts.to_prob(), 0.42)

    def test_bitstream_to_bitstream(self) -> None:
        """A bitstream tensor converts to itself."""
        bits = np.array([[1, 0, 1]], dtype=np.uint8)
        ts = TensorStream(data=bits, domain="bitstream")
        np.testing.assert_array_equal(ts.to_bitstream(), bits)

    def test_quantum_to_quantum(self) -> None:
        """A quantum tensor converts to itself."""
        q = np.array([[0.6 + 0j, 0.8 + 0j]])
        ts = TensorStream(data=q, domain="quantum")
        np.testing.assert_array_equal(ts.to_quantum(), q)
