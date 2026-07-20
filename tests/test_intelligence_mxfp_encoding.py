# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — MXFP encoding contracts

"""Contracts for microscaling floating-point block encoding."""

from __future__ import annotations

import pytest


class TestMXFP:
    """Tests for MXFP / Block-FP encoding/decoding."""

    def test_mxfp4_config(self) -> None:
        """MXFP4 config matches OCP spec."""
        from sc_neurocore.compiler.intelligence import MXFP4

        assert MXFP4.element_bits == 4
        assert MXFP4.block_size == 32
        assert MXFP4.shared_exp_bits == 8
        assert MXFP4.label == "MXFP4"
        assert MXFP4.bits_per_block == 8 + 32 * 4  # 136

    def test_mxfp8_e4m3_config(self) -> None:
        """MXFP8 E4M3 config."""
        from sc_neurocore.compiler.intelligence import MXFP8_E4M3

        assert MXFP8_E4M3.element_bits == 8
        assert MXFP8_E4M3.exp_bits == 4
        assert MXFP8_E4M3.mantissa_bits == 3

    def test_fp8_no_shared_exp(self) -> None:
        """IEEE FP8 has no shared exponent (block_size=1)."""
        from sc_neurocore.compiler.intelligence import FP8_E4M3

        assert FP8_E4M3.block_size == 1
        assert FP8_E4M3.shared_exp_bits == 0

    def test_encode_decode_roundtrip_mxfp4(self) -> None:
        """MXFP4 encode→decode roundtrip preserves sign and order."""
        from sc_neurocore.compiler.intelligence import (
            MXFP4,
            mxfp_encode_block,
            mxfp_decode_block,
        )

        values = [float(i) / 32 for i in range(32)]
        exp, encoded = mxfp_encode_block(values, MXFP4)
        decoded = mxfp_decode_block(exp, encoded, MXFP4)
        # Order preserved
        for i in range(1, len(decoded)):
            assert decoded[i] >= decoded[i - 1]

    def test_encode_all_zeros(self) -> None:
        """All-zero block returns zero exponent."""
        from sc_neurocore.compiler.intelligence import (
            MXFP4,
            mxfp_encode_block,
        )

        exp, encoded = mxfp_encode_block([0.0] * 32, MXFP4)
        assert exp == 0
        assert all(e == 0 for e in encoded)

    def test_block_size_mismatch_raises(self) -> None:
        """Wrong block size raises ValueError."""
        from sc_neurocore.compiler.intelligence import (
            MXFP4,
            mxfp_encode_block,
        )

        with pytest.raises(ValueError, match="Block size"):
            mxfp_encode_block([1.0, 2.0], MXFP4)

    def test_negative_values(self) -> None:
        """Negative values have sign bit set."""
        from sc_neurocore.compiler.intelligence import (
            MXFP4,
            mxfp_encode_block,
            mxfp_decode_block,
        )

        values = [-1.0] * 32
        exp, encoded = mxfp_encode_block(values, MXFP4)
        decoded = mxfp_decode_block(exp, encoded, MXFP4)
        assert all(d < 0 for d in decoded)

    def test_mxfp6_exists(self) -> None:
        """MXFP6 config exists."""
        from sc_neurocore.compiler.intelligence import MXFP6

        assert MXFP6.element_bits == 6
