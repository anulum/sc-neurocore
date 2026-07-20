# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wave 6 compiler integration contracts

"""Named workflow contracts for the Wave 6 compiler feature integration."""

from __future__ import annotations


class TestWave6Integration:
    """Cross-feature integration tests."""

    def test_provenance_then_compliance(self) -> None:
        """Provenance chain enables compliance coverage."""
        from sc_neurocore.compiler.intelligence import (
            generate_provenance_chain,
            generate_compliance_matrix,
        )

        chain = generate_provenance_chain("sc_lif", {"v": "a + b"})
        assert len(chain) == 3
        matrix = generate_compliance_matrix(
            "sc_lif",
            has_provenance=True,
            has_tmr=True,
            has_checksum=True,
            has_sva=True,
        )
        all_covered = all(e.status == "covered" for e in matrix)
        assert all_covered

    def test_timescale_then_dispatch(self) -> None:
        """Partition timescales, then dispatch to backends."""
        from sc_neurocore.compiler.intelligence import (
            partition_timescales,
            plan_heterogeneous_dispatch,
        )

        part = partition_timescales(
            {"v": "a + b", "w": "c + d"},
            time_constants={"v": 1.0, "w": 100.0},
        )
        all_eqs = {**part.fast_equations, **part.slow_equations}
        plan = plan_heterogeneous_dispatch(
            all_eqs,
            ["fpga", "mcu"],
        )
        assert plan.estimated_speedup > 1.0

    def test_equivalence_then_lint(self) -> None:
        """Generate proof sketch, then lint for side channels."""
        from sc_neurocore.compiler.intelligence import (
            generate_equivalence_sketch,
            lint_side_channels,
        )

        sketch = generate_equivalence_sketch(
            "sc_hh",
            {"v": "a * b / c + d"},
        )
        findings = lint_side_channels({"v": "a * b / c + d"})
        assert sketch.quantisation_bound > 0
        assert len(findings) >= 3  # div + mul + spike

    def test_energy_schedule_for_mcu(self) -> None:
        """Energy schedule on edge MCU profile."""
        from sc_neurocore.compiler.platforms import get_profile
        from sc_neurocore.compiler.intelligence import (
            generate_energy_schedule,
        )

        p = get_profile("esp32_s3")
        assert p.platform_class == "edge_mcu"
        s = generate_energy_schedule(500, energy_budget_uj=5.0)
        assert s.neurons_per_epoch <= 500
