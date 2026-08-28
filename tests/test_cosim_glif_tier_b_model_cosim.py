# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTierBModelCosim from former test_cosim_glif.py

"""Focused suite: TestTierBModelCosim from former test_cosim_glif.py."""

from __future__ import annotations

from tests.cosim_glif_support import *  # noqa: F403


_ModelType = type[GLIFNeuron] | type[SCFourStateGLIFNeuron]


class TestTierBModelCosim:
    """WC-A5 Tier-B GLIF schema enrolment."""

    @pytest.mark.parametrize(
        ("schema_name", "model_type", "variables", "expected_events", "tolerance"),
        [
            (
                "glif",
                GLIFNeuron,
                (
                    "v",
                    "theta_spike",
                    "i_asc1",
                    "i_asc2",
                    "theta_voltage",
                    "refractory_remaining",
                ),
                167,
                1e-12,
            ),
            (
                "sc_four_state_glif",
                SCFourStateGLIFNeuron,
                ("v", "theta", "i_asc1", "i_asc2"),
                181,
                0.0,
            ),
        ],
    )
    def test_paired_schemas_match_public_step_sequence(
        self,
        schema_name: str,
        model_type: _ModelType,
        variables: tuple[str, ...],
        expected_events: int,
        tolerance: float,
    ) -> None:
        """Both paired schemas reproduce every public state and event.

        The 4,000-step varied drive exercises the canonical GLIF5 exact flow and
        retained four-state RK4 identity independently. Exact state and event
        equality catches schema drift in either identity.
        """
        schema_dir = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
        hand = model_type()
        toml_schema = UniversalNeuron.from_schema(schema_dir / f"{schema_name}.toml")
        json_schema = UniversalNeuron.from_schema(schema_dir / f"{schema_name}.json")
        currents = (0.0, 15.0, 22.0, 30.0, 45.0, 50.0, 30.0, 22.0) * 500
        spike_count = 0
        reset_count = 0

        for current in currents:
            hand_spike = hand.step(current)
            spike_count += hand_spike
            if hand_spike:
                assert hand.v == getattr(hand, "e_l", getattr(hand, "v_reset", None))
                reset_count += 1
            assert int(bool(toml_schema.step(I=current))) == hand_spike
            assert int(bool(json_schema.step(I=current))) == hand_spike
            for variable in variables:
                expected = getattr(hand, variable)
                assert toml_schema.state[variable] == pytest.approx(expected, abs=tolerance)
                assert json_schema.state[variable] == pytest.approx(expected, abs=tolerance)

        assert spike_count == reset_count == expected_events
