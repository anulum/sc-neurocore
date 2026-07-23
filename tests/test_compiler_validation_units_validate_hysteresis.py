# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestValidateHysteresis from former test_compiler_validation_units.py

"""Focused suite: TestValidateHysteresis from former test_compiler_validation_units.py."""

from __future__ import annotations

from tests.compiler_validation_units_support import *  # noqa: F403

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
