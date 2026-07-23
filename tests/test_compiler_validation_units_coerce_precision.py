# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCoercePrecision from former test_compiler_validation_units.py

"""Focused suite: TestCoercePrecision from former test_compiler_validation_units.py."""

from __future__ import annotations

from tests.compiler_validation_units_support import *  # noqa: F403

class TestCoercePrecision:
    """Each precision input resolves the expected datapath parameters and tags
    its manifest with the matching provenance and kind."""

    def test_none_falls_back_to_default_qformat(self) -> None:
        width, fraction, label, manifest, fmt = _coerce_precision(
            None, default_width=16, default_frac=8, tag="weights"
        )
        assert (width, fraction, label) == (16, 8, "Q8.8")
        assert isinstance(fmt, QFormat)
        assert manifest["source"] == "weights:fallback"
        assert manifest["kind"] == "fixed"

    def test_qformat_string_resolves_total_and_fraction(self) -> None:
        width, fraction, label, manifest, fmt = _coerce_precision(
            "Q16.16", default_width=16, default_frac=8, tag="state"
        )
        assert (width, fraction) == (32, 16)
        assert isinstance(fmt, QFormat)
        assert manifest["source"] == "Q16.16"
        assert manifest["kind"] == "fixed"

    def test_block_floating_string_resolves_mantissa_and_passes_parameter_count(self) -> None:
        width, fraction, label, manifest, fmt = _coerce_precision(
            "BFP16E3X32", default_width=16, default_frac=8, tag="weights", parameter_count=5
        )
        assert (width, fraction) == (16, 15)
        assert isinstance(fmt, BlockFloatingMode)
        assert manifest["source"] == "BFP16E3X32"
        assert manifest["kind"] == "block_floating"
        assert manifest["parameter_count"] == 5

    def test_unparsable_format_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a fixed Q-format or block-floating format"):
            _coerce_precision("not-a-format", default_width=16, default_frac=8, tag="weights")
