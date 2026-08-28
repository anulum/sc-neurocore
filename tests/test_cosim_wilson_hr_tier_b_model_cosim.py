# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wilson-HR schema and reference-integrator parity

"""Exercise both Wilson-HR schema formats against an independent RK4 path."""

from __future__ import annotations

from tests.cosim_wilson_hr_support import *


class TestTierBModelCosim:
    """WC-A5 Tier-B Wilson-HR schema enrolment."""

    def test_wilson_hr_schema_formats_match_hand_rk4_sequence(self) -> None:
        """The TOML and JSON schemas track Wilson-HR over a varied drive.

        Five passes through eight 100-step drive blocks exercise source capacitance,
        the polynomial membrane nullcline, coupled recovery flow, all four RK4 stages,
        and continuous threshold crossings. Both formats must reproduce every event
        decision and both post-step states exactly without a reset discontinuity.
        """
        schema_dir = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
        hand = WilsonHRNeuron()
        toml_schema = UniversalNeuron.from_schema(schema_dir / "wilson_hr.toml")
        json_schema = UniversalNeuron.from_schema(schema_dir / "wilson_hr.json")
        current_blocks = (0.0, 0.075, 0.1, 0.14, 0.0, 0.05, 0.1, 0.075)
        spike_count = 0

        for _cycle in range(5):
            for current in current_blocks:
                for _step in range(100):
                    hand_spike = hand.step(current)
                    spike_count += hand_spike
                    if hand_spike:
                        assert hand.v >= hand.v_peak
                    assert int(bool(toml_schema.step(I=current))) == hand_spike
                    assert int(bool(json_schema.step(I=current))) == hand_spike
                    for variable in ("v", "r"):
                        expected = getattr(hand, variable)
                        assert toml_schema.state[variable] == expected
                        assert json_schema.state[variable] == expected

        assert spike_count == 25
