# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQ88 from former test_gen_vmin_lif_lut.py

"""Focused suite: TestQ88 from former test_gen_vmin_lif_lut.py."""

from __future__ import annotations

import pytest
from tools.gen_vmin_lif_lut import (
    decode_q88,
    encode_q88,
    Q88_MAX,
    Q88_MIN,
    Q88_SCALE,
)


class TestQ88:
    """Encode and decode signed Q8.8 fixed point."""

    def test_encode_zero(self) -> None:
        """Zero encodes to zero."""
        assert encode_q88(0.0) == 0

    def test_encode_one(self) -> None:
        """The value one encodes to 256."""
        assert encode_q88(1.0) == 256

    def test_encode_negative_one(self) -> None:
        """Minus one encodes to -256."""
        assert encode_q88(-1.0) == -256

    def test_encode_quarter(self) -> None:
        """A quarter encodes to 64."""
        assert encode_q88(0.25) == 64

    def test_encode_max_clamp(self) -> None:
        """A huge positive value clamps to the Q8.8 maximum."""
        assert encode_q88(1e9) == Q88_MAX

    def test_encode_min_clamp(self) -> None:
        """A huge negative value clamps to the Q8.8 minimum."""
        assert encode_q88(-1e9) == Q88_MIN

    def test_decode_round_trip(self) -> None:
        """Decoding an encoded value returns it within one Q8.8 step."""
        for v in [0.0, 0.25, 0.5, 1.0, -1.0, 5.5, -5.0]:
            assert decode_q88(encode_q88(v)) == pytest.approx(v, abs=1.0 / 256)

    def test_q88_scale_constant(self) -> None:
        """The Q8.8 scale is 256."""
        assert Q88_SCALE == 256
