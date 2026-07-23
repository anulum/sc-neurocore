# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPrecisionFormatParser from former test_quantizer.py

"""Focused suite: TestPrecisionFormatParser from former test_quantizer.py."""

from __future__ import annotations

from tests.quantizer_support import *  # noqa: F403

class TestPrecisionFormatParser:
    """Test format parsing for fixed and block-floating modes."""

    def test_parse_block_floating_alias(self):
        fmt = parse_precision_format("BFP16E3X32")
        assert isinstance(fmt, BlockFloatingMode)
        assert fmt.mantissa_bits == 16
        assert fmt.exponent_bits == 3
        assert fmt.block_size == 32

    def test_parse_block_floating_flexible_alias(self):
        fmt = parse_precision_format("bfp16_e3")
        assert isinstance(fmt, BlockFloatingMode)
        assert fmt.mantissa_bits == 16
        assert fmt.exponent_bits == 3

    def test_parse_block_floating_dash_alias(self):
        fmt = parse_precision_format("BFP16-3x32")
        assert isinstance(fmt, BlockFloatingMode)
        assert fmt.block_size == 32

    def test_parse_rejects_non_string_format(self):
        with pytest.raises(TypeError, match="precision format string"):
            parse_precision_format(123)  # type: ignore[arg-type]

    def test_precision_label_appends_block_size_when_source_omits_it(self):
        # When the originating string lacks the explicit `X<block_size>` suffix
        # the telemetry label re-attaches it so block-floating modes stay
        # unambiguous.
        from sc_neurocore.compiler.manifest_gen import _precision_label

        parsed = parse_precision_format("BFP16E3X32")
        label = _precision_label(parsed, source="BFP16E3")
        assert label.endswith(f"X{parsed.block_size}")
        # The already-suffixed source is returned verbatim.
        assert not _precision_label(parsed, source="BFP16E3X32").endswith("X32X32")
