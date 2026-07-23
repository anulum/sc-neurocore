# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPow from former test_c_expr_emitter.py

"""Focused suite: TestPow from former test_c_expr_emitter.py."""

from __future__ import annotations

from tests.c_expr_emitter_support import *  # noqa: F403

class TestPow:
    def test_square_is_repeated_multiply(self) -> None:
        assert _emit("v ** 2", {"v"}) == "(v * v)"

    def test_cube(self) -> None:
        assert _emit("v ** 3", {"v"}) == "(v * v * v)"

    def test_eighth_power(self) -> None:
        assert _emit("v ** 8", {"v"}).count("v *") == 7

    def test_sqrt_from_half_power(self) -> None:
        assert _emit("v ** 0.5", {"v"}) == "hls::sqrt(v)"

    def test_cbrt_from_third_power(self) -> None:
        assert _emit("v ** (1.0 / 3.0)", {"v"}) == "hls::cbrt(v)"

    def test_unsupported_power_raises(self) -> None:
        with pytest.raises(ValueError, match="Only integer powers"):
            _emit("v ** 9", {"v"})
