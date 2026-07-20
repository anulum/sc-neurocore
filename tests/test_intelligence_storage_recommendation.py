# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Storage recommendation contracts

"""Contracts for compiler storage recommendation."""

from __future__ import annotations


class TestStorageRecommendation:
    """Tests for BRAM/register storage strategy."""

    def test_small_uses_registers(self) -> None:
        """≤64 neurons → registers."""
        from sc_neurocore.compiler.intelligence import storage_recommendation

        rec = storage_recommendation(32, 16)
        assert rec.strategy == "registers"
        assert rec.total_bits == 32 * 16

    def test_medium_uses_bram(self) -> None:
        """65–16K neurons → BRAM."""
        from sc_neurocore.compiler.intelligence import storage_recommendation

        rec = storage_recommendation(1024, 16)
        assert rec.strategy == "bram"
        assert rec.total_bits == 1024 * 16

    def test_large_with_uram(self) -> None:
        """≥16K neurons with URAM → URAM."""
        from sc_neurocore.compiler.intelligence import storage_recommendation

        rec = storage_recommendation(20000, 16, has_uram=True)
        assert rec.strategy == "uram"
        assert rec.uram_used >= 1

    def test_large_without_uram_uses_bram(self) -> None:
        """Large without URAM → falls back to BRAM."""
        from sc_neurocore.compiler.intelligence import storage_recommendation

        rec = storage_recommendation(20000, 16, has_uram=False)
        assert rec.strategy == "bram"

    def test_custom_threshold(self) -> None:
        """Custom register threshold."""
        from sc_neurocore.compiler.intelligence import storage_recommendation

        rec = storage_recommendation(100, 16, register_threshold=128)
        assert rec.strategy == "registers"

    def test_bram_18k_for_small(self) -> None:
        """Small BRAM uses 18Kb tile."""
        from sc_neurocore.compiler.intelligence import storage_recommendation

        rec = storage_recommendation(128, 16)  # 2048 bits, fits in 18Kb
        assert rec.strategy == "bram"
        assert rec.bram_18k_used == 1
        assert rec.bram_36k_used == 0

    def test_bram_36k_for_large(self) -> None:
        """Larger BRAM uses 36Kb tiles."""
        from sc_neurocore.compiler.intelligence import storage_recommendation

        rec = storage_recommendation(4096, 16)  # 65536 bits
        assert rec.strategy == "bram"
        assert rec.bram_36k_used >= 1

    def test_reason_populated(self) -> None:
        """Reason string is non-empty."""
        from sc_neurocore.compiler.intelligence import storage_recommendation

        rec = storage_recommendation(10, 16)
        assert len(rec.reason) > 0
