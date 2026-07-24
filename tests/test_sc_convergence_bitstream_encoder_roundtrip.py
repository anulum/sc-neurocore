# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBitstreamEncoderRoundtrip from former test_sc_convergence.py

"""Focused suite: TestBitstreamEncoderRoundtrip from former test_sc_convergence.py."""

from __future__ import annotations

from tests.sc_convergence_support import *  # noqa: F403


class TestBitstreamEncoderRoundtrip:
    """BitstreamEncoder.encode → bitstream_to_probability roundtrip."""

    @pytest.mark.parametrize("p", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    def test_roundtrip_accuracy(self, p):
        enc = BitstreamEncoder(x_min=0.0, x_max=1.0, length=4096, seed=42)
        bits = enc.encode(p)
        recovered = bitstream_to_probability(bits)
        np.testing.assert_allclose(recovered, p, atol=0.03)

    def test_output_is_binary(self):
        enc = BitstreamEncoder(x_min=0.0, x_max=1.0, length=1024, seed=42)
        bits = enc.encode(0.6)
        assert set(np.unique(bits)).issubset({0, 1})

    def test_output_length(self):
        enc = BitstreamEncoder(x_min=0.0, x_max=1.0, length=777, seed=42)
        bits = enc.encode(0.5)
        assert len(bits) == 777
