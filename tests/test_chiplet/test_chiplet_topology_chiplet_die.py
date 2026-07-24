# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestChipletDie from former test_chiplet_topology.py

"""Focused suite: TestChipletDie from former test_chiplet_topology.py."""

from __future__ import annotations

from chiplet_topology_support import *  # noqa: F403


class TestChipletDie:
    """Die timing, seed, and width contracts."""

    def test_defaults_and_clock_period(self) -> None:
        die = ChipletDie(die_id=0, clock_mhz=100.0)
        assert die.clock_period_ns == 10.0
        assert die.n_neurons == 128

    def test_custom_seed_is_preserved(self) -> None:
        assert ChipletDie(die_id=5, lfsr_seed=0xBEEF).lfsr_seed == 0xBEEF

    @pytest.mark.parametrize(
        "constructor",
        [
            lambda: ChipletDie(-1),
            lambda: ChipletDie(0, clock_mhz=0.0),
            lambda: ChipletDie(0, lfsr_seed=0),
            lambda: ChipletDie(0, n_neurons=0),
        ],
    )
    def test_invalid_die_contracts_fail(self, constructor: Callable[[], object]) -> None:
        with pytest.raises(ValueError):
            constructor()
