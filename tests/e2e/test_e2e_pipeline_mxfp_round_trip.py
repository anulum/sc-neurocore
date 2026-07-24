# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMXFPRoundTrip from former test_e2e_pipeline.py

"""Focused suite: TestMXFPRoundTrip from former test_e2e_pipeline.py."""

from __future__ import annotations

from tests.e2e.e2e_pipeline_support import *  # noqa: F403


@pytest.mark.e2e
class TestMXFPRoundTrip:
    """Block-FP encode → decode preserves values within precision bounds."""

    def test_mxfp8_e4m3_round_trip(self):
        """MXFP8 E4M3: encode → decode → sign preservation."""
        from sc_neurocore.compiler.intelligence import (
            MXFP8_E4M3,
            mxfp_encode_block,
            mxfp_decode_block,
        )

        # Block size is 32, so we need exactly 32 values
        values = [0.0, 1.0, -1.0, 0.5] * 8  # 32 values
        shared_exp, elements = mxfp_encode_block(values, MXFP8_E4M3)
        decoded = mxfp_decode_block(shared_exp, elements, MXFP8_E4M3)

        assert len(decoded) == len(values)
        for orig, dec in zip(values, decoded):
            if orig != 0:
                assert (orig > 0) == (dec > 0), f"Sign flip: {orig} → {dec}"

    def test_zero_stability(self):
        """Zero encodes and decodes as zero."""
        from sc_neurocore.compiler.intelligence import (
            MXFP8_E5M2,
            mxfp_encode_block,
            mxfp_decode_block,
        )

        values = [0.0] * 32  # Block size = 32
        shared_exp, elements = mxfp_encode_block(values, MXFP8_E5M2)
        decoded = mxfp_decode_block(shared_exp, elements, MXFP8_E5M2)
        assert all(d == 0.0 for d in decoded)

    def test_all_configs_round_trip(self):
        """Every block-FP config can encode/decode without crashing."""
        from sc_neurocore.compiler.intelligence import (
            MXFP4,
            MXFP6,
            MXFP8_E4M3,
            MXFP8_E5M2,
            mxfp_encode_block,
            mxfp_decode_block,
        )

        # Only test block-FP configs (shared_exp_bits > 0)
        # FP8 standalone (block_size=1) uses per-element exponent
        configs = {
            "MXFP4": MXFP4,
            "MXFP6": MXFP6,
            "MXFP8_E4M3": MXFP8_E4M3,
            "MXFP8_E5M2": MXFP8_E5M2,
        }
        for name, config in configs.items():
            values = ([1.0, -1.0, 0.5, 0.0] * max(1, config.block_size // 4))[: config.block_size]
            shared_exp, elements = mxfp_encode_block(values, config)
            decoded = mxfp_decode_block(shared_exp, elements, config)
            assert len(decoded) == len(values), f"{name}: length mismatch"
