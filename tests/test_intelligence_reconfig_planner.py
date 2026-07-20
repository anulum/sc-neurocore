# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Partial-reconfiguration planner contracts

"""Contracts for compiler partial-reconfiguration planning."""

from __future__ import annotations


class TestPartialReconfig:
    def test_basic_plan(self) -> None:
        from sc_neurocore.compiler.intelligence import (
            plan_partial_reconfiguration,
        )

        plan = plan_partial_reconfiguration({"v": "a", "u": "b"})
        assert plan.total_regions == 2
        assert plan.bitstream_count == 2
        assert len(plan.schedule) == 2

    def test_single_var(self) -> None:
        from sc_neurocore.compiler.intelligence import (
            plan_partial_reconfiguration,
        )

        plan = plan_partial_reconfiguration({"v": "a"})
        assert plan.total_regions == 1

    def test_custom_slots(self) -> None:
        from sc_neurocore.compiler.intelligence import (
            plan_partial_reconfiguration,
        )

        plan = plan_partial_reconfiguration(
            {"v": "a", "u": "b", "w": "c"},
            time_slots=4,
        )
        assert plan.bitstream_count == 4
