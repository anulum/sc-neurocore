# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBlockExponentLayout from former test_quantizer.py

"""Focused suite: TestBlockExponentLayout from former test_quantizer.py."""

from __future__ import annotations

from tests.quantizer_support import *  # noqa: F403


class TestBlockExponentLayout:
    """Validate explicit block-exponent metadata for BFP compiler surfaces."""

    def test_block_exponent_layout_computes_partial_tail(self):
        mode = BlockFloatingMode.from_aliases("BFP16E3X32")
        layout = mode.block_exponent_layout(65)

        assert mode.block_exponent_count(65) == 3
        assert layout.manifest() == {
            "alignment": "contiguous_flattened_block",
            "flattened_order": "row_major",
            "parameter_count": 65,
            "block_size": 32,
            "exponent_count": 3,
            "last_block_size": 1,
            "exponent_index_formula": "parameter_index // block_size",
        }

    def test_block_exponent_layout_rejects_bad_counts_and_vectors(self):
        mode = BlockFloatingMode.from_aliases("BFP16E3X32")

        with pytest.raises(ValueError, match="non-negative"):
            mode.block_exponent_count(-1)
        with pytest.raises(TypeError, match="integer"):
            mode.block_exponent_count(True)
        with pytest.raises(ValueError, match="exponent count mismatch"):
            mode.validate_exponents(np.array([0, 1], dtype=np.int64), parameter_count=65)
        with pytest.raises(ValueError, match="configured block-floating range"):
            mode.validate_exponents(np.array([0, 8, 1], dtype=np.int64), parameter_count=65)
        with pytest.raises(TypeError, match="integer codes"):
            mode.validate_exponents(np.array([0.0, 1.0, 2.0]), parameter_count=65)
