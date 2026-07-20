# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Compiler auto-quantisation contracts

"""Contracts for compiler auto-quantisation design-space exploration."""

from __future__ import annotations


class TestQuantisationSweep:
    """Quantisation design-space exploration."""

    def test_default_sweep(self) -> None:
        from sc_neurocore.compiler.intelligence import auto_quantisation_sweep

        results = auto_quantisation_sweep({"v": "a * b + c"})
        assert len(results) == 7  # [4, 8, 12, 16, 20, 24, 32]

    def test_widths_sorted(self) -> None:
        from sc_neurocore.compiler.intelligence import auto_quantisation_sweep

        results = auto_quantisation_sweep({"v": "a + b"})
        widths = [r.data_width for r in results]
        assert widths == sorted(widths)

    def test_luts_grow_with_width(self) -> None:
        from sc_neurocore.compiler.intelligence import auto_quantisation_sweep

        results = auto_quantisation_sweep({"v": "a + b + c"})
        luts = [r.estimated_luts for r in results]
        assert luts == sorted(luts)  # Monotonically increasing

    def test_precision_improves_with_width(self) -> None:
        from sc_neurocore.compiler.intelligence import auto_quantisation_sweep

        results = auto_quantisation_sweep({"v": "a * b"})
        steps = [r.min_step for r in results]
        assert steps == sorted(steps, reverse=True)  # Smaller step = better

    def test_custom_widths(self) -> None:
        from sc_neurocore.compiler.intelligence import auto_quantisation_sweep

        results = auto_quantisation_sweep(
            {"v": "a + b"},
            widths=[8, 16, 32],
        )
        assert len(results) == 3

    def test_format_report(self) -> None:
        from sc_neurocore.compiler.intelligence import (
            auto_quantisation_sweep,
            format_quantisation_report,
        )

        results = auto_quantisation_sweep({"v": "a * b + c"})
        report = format_quantisation_report(results)
        assert "Q-format" in report
        assert "LUTs" in report
        assert "LSB Step" in report

    def test_target_affects_dsps(self) -> None:
        """Targets with DSP blocks should show DSP usage."""
        from sc_neurocore.compiler.intelligence import auto_quantisation_sweep

        r_artix = auto_quantisation_sweep(
            {"v": "a * b"},
            target="artix7",
        )
        r_ice40 = auto_quantisation_sweep(
            {"v": "a * b"},
            target="bae_rad750",
        )
        # Artix has DSP48E1, RAD750 doesn't
        assert all(r.estimated_dsps > 0 for r in r_artix)
        assert all(r.estimated_dsps == 0 for r in r_ice40)

    def test_izh_multi_equation(self) -> None:
        from sc_neurocore.compiler.intelligence import auto_quantisation_sweep

        results = auto_quantisation_sweep(
            {
                "v": "0.04 * v * v + 5 * v + 140 - u + I",
                "u": "a * (b * v - u)",
            }
        )
        # More equations → more FFs
        for r in results:
            assert r.estimated_ffs == 2 * r.data_width  # 2 state vars
