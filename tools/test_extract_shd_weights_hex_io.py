# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHexIO from former test_extract_shd_weights.py

"""Focused suite: TestHexIO from former test_extract_shd_weights.py."""

from __future__ import annotations

from extract_shd_weights_support import *  # noqa: F403

class TestHexIO:
    def test_write_signed_int8_hex_round_trip(self, tmp_path: Path) -> None:
        w = torch.tensor([[-128, -1, 0, 1, 127]], dtype=torch.int8)
        path = str(tmp_path / "w.hex")
        write_int8_hex(w, path)
        with open(path) as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("//")]
        # -128 → 0x80, -1 → 0xff, 0 → 0x00, 1 → 0x01, 127 → 0x7f
        assert lines == ["80", "ff", "00", "01", "7f"]

    def test_write_delays_negative_range(self, tmp_path: Path) -> None:
        delays = [-15, -1, 0, 1, 15]
        path = str(tmp_path / "d.hex")
        write_delays_hex(delays, path)
        with open(path) as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("//")]
        # -15 → -15+256 = 241 = 0xf1, -1 → 0xff, 0 → 0x00, 1 → 0x01, 15 → 0x0f
        assert lines == ["f1", "ff", "00", "01", "0f"]
