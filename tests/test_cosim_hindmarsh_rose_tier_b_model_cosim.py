# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTierBModelCosim from former test_cosim_hindmarsh_rose.py

"""Focused suite: TestTierBModelCosim from former test_cosim_hindmarsh_rose.py."""

from __future__ import annotations

from tests.cosim_hindmarsh_rose_support import *  # noqa: F403

class TestTierBModelCosim:
    """WC-A5 Tier-B Hindmarsh-Rose schema enrolment."""

    def test_hindmarsh_rose_schema_formats_match_hand_rk4_sequence(self) -> None:
        """Both schemas reproduce the maintained no-reset RK4 hand model.

        The 1,200-step drive alternates silent, depolarising, and negative currents.
        It exercises every RK4 stage in the three coupled equations, three upward
        ``x >= x_threshold`` crossings, and three subsequent below-threshold re-arms.
        TOML and JSON must agree exactly; their states must remain within ``1e-10`` of
        the independently implemented hand integrator despite harmless operation-order
        rounding at float64 precision.
        """
        schema_dir = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
        hand = HindmarshRoseNeuron()
        toml_schema = UniversalNeuron.from_schema(schema_dir / "hindmarsh_rose.toml")
        json_schema = UniversalNeuron.from_schema(schema_dir / "hindmarsh_rose.json")
        currents = (0.0, 2.0, 3.0, 5.0, 4.0, -1.0) * 200
        spike_count = 0
        rearm_count = 0
        was_above = hand.x >= hand.x_threshold

        for current in currents:
            hand_spike = hand.step(current)
            toml_spike = toml_schema.step(I=current)
            json_spike = json_schema.step(I=current)
            spike_count += hand_spike
            now_above = hand.x >= hand.x_threshold
            if was_above and not now_above:
                rearm_count += 1
            was_above = now_above

            assert toml_spike == json_spike == hand_spike
            for variable in ("x", "y", "z"):
                assert toml_schema.state[variable] == json_schema.state[variable]
                assert toml_schema.state[variable] == pytest.approx(
                    getattr(hand, variable), abs=1e-10, rel=1e-12
                )

        assert spike_count == 3
        assert rearm_count == 3
