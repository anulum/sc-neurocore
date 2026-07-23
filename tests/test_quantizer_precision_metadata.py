# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPrecisionMetadata from former test_quantizer.py

"""Focused suite: TestPrecisionMetadata from former test_quantizer.py."""

from __future__ import annotations

from tests.quantizer_support import *  # noqa: F403

class TestPrecisionMetadata:
    """Validate precision-format parse coverage for wide fixed-point formats."""

    def test_parse_q16_16(self):
        q = QFormat.from_string("Q16.16")
        assert q.integer_bits == 16
        assert q.fraction_bits == 16
        assert q.total_bits == 32
        assert q.scale == 65536

    def test_public_precision_constants(self):
        assert QFormat.from_string("Q8.8") == Q8_8
        assert QFormat.from_string("Q16.16") == Q16_16
