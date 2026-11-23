# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWatermark from former test_intelligence_security_and_compliance.py

"""Focused suite: TestWatermark from former test_intelligence_security_and_compliance.py."""

from __future__ import annotations


class TestWatermark:
    """Watermarking the design."""

    def test_basic(self) -> None:
        """The watermark is verifiable with little overhead."""
        from sc_neurocore.compiler.intelligence import embed_watermark

        r = embed_watermark("sc_lif", {"v": "a"})
        assert r.verifiable is True
        assert len(r.watermark_hash) == 16
        assert r.overhead_percent <= 1.0

    def test_deterministic(self) -> None:
        """The same owner gives the same watermark."""
        from sc_neurocore.compiler.intelligence import embed_watermark

        r1 = embed_watermark("sc_lif", {"v": "a"}, owner_id="Lab1")
        r2 = embed_watermark("sc_lif", {"v": "a"}, owner_id="Lab1")
        assert r1.watermark_hash == r2.watermark_hash

    def test_different_owners(self) -> None:
        """Different owners give different watermarks."""
        from sc_neurocore.compiler.intelligence import embed_watermark

        r1 = embed_watermark("sc_lif", {"v": "a"}, owner_id="Lab1")
        r2 = embed_watermark("sc_lif", {"v": "a"}, owner_id="Lab2")
        assert r1.watermark_hash != r2.watermark_hash
