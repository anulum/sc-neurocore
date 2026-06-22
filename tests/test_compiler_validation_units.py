# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Unit contracts for dual-datapath precision validation

"""Branch-level contracts for adaptive-runtime precision coercion and the
low-precision / high-precision datapath and hysteresis validators."""

from __future__ import annotations

import math

import pytest

from sc_neurocore.compiler.quantizer import BlockFloatingMode, QFormat
from sc_neurocore.compiler.validation import (
    _coerce_precision,
    _validate_hysteresis,
    _validate_lp_hp,
)


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


class TestValidateLpHp:
    """The low/high-precision pair must order widths, keep both fractions
    positive, and leave the LP datapath at least two bits wide."""

    def test_accepts_sensible_pair(self) -> None:
        _validate_lp_hp(4, 2, 8, 4)  # a well-ordered pair must validate silently

    def test_rejects_lp_width_not_below_hp(self) -> None:
        with pytest.raises(ValueError, match="must be strictly less than HP data_width"):
            _validate_lp_hp(8, 2, 8, 4)

    def test_rejects_lp_fraction_below_one(self) -> None:
        with pytest.raises(ValueError, match=r"LP fraction \(0\) must be >= 1"):
            _validate_lp_hp(4, 0, 8, 4)

    def test_rejects_hp_fraction_below_one(self) -> None:
        with pytest.raises(ValueError, match=r"HP fraction \(0\) must be >= 1"):
            _validate_lp_hp(4, 2, 8, 0)

    def test_rejects_lp_width_below_two(self) -> None:
        # lp_width=1 is below hp_width=8 and both fractions are positive, so the
        # final minimum-width guard is the one that fires.
        with pytest.raises(ValueError, match=r"LP data_width \(1\) must be >= 2"):
            _validate_lp_hp(1, 1, 8, 4)


class TestValidateHysteresis:
    """Hysteresis thresholds must be finite, strictly ordered inside the unit
    interval, and quantise to a usable 1 <= down < up < max_lp_code range."""

    def test_accepts_valid_thresholds(self) -> None:
        _validate_hysteresis(0.5, 0.2, max_lp_code=16)  # ordered thresholds validate silently

    def test_rejects_non_finite_threshold(self) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            _validate_hysteresis(math.inf, 0.2, max_lp_code=16)

    def test_rejects_up_outside_unit_interval(self) -> None:
        with pytest.raises(ValueError, match="threshold_up_pct must satisfy"):
            _validate_hysteresis(1.5, 0.2, max_lp_code=16)

    def test_rejects_down_not_below_up(self) -> None:
        with pytest.raises(ValueError, match="threshold_down_pct must satisfy"):
            _validate_hysteresis(0.5, 0.6, max_lp_code=16)

    def test_rejects_degenerate_quantised_codes(self) -> None:
        # Valid float thresholds, but a tiny max_lp_code collapses both quantised
        # codes so the 1 <= down < up < max_lp_code ordering cannot hold.
        with pytest.raises(ValueError, match="Quantised threshold codes must satisfy"):
            _validate_hysteresis(0.5, 0.2, max_lp_code=2)
