# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Pernarowski schema and hand-model parity

"""Compare Pernarowski schema dynamics with the reference hand model."""

from __future__ import annotations

from tests.cosim_pernarowski_support import *  # noqa: F403


class TestSchemaModelParity:
    """Pernarowski schema parity."""

    def test_pernarowski_schema_formats_match_hand_rk4_sequence(self) -> None:
        """The TOML and JSON schemas reproduce the hand model over a varied drive.

        The 5,000-step sequence exercises the external-current term and every RK4
        stage across the fast cubic coordinate, recovery variable, and ultra-slow
        adaptation variable. It also covers 17 upward crossings and 17 subsequent
        below-threshold re-arms. Exact state equality is required after every step,
        so either schema format drifting in initial state, parameters, equations,
        operation order, or no-reset edge detection fails immediately.
        """
        schema_dir = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
        hand = PernarowskiNeuron()
        toml_schema = UniversalNeuron.from_schema(schema_dir / "pernarowski.toml")
        json_schema = UniversalNeuron.from_schema(schema_dir / "pernarowski.json")
        currents = (0.0, 0.1, -0.1, 0.2, 0.0, -0.2, 0.15, 0.05) * 625
        spike_count = 0
        rearm_count = 0
        was_above = hand.v >= hand.v_threshold

        for current in currents:
            hand_spike = hand.step(current)
            spike_count += hand_spike
            now_above = hand.v >= hand.v_threshold
            if was_above and not now_above:
                rearm_count += 1
            was_above = now_above
            assert int(bool(toml_schema.step(I=current))) == hand_spike
            assert int(bool(json_schema.step(I=current))) == hand_spike
            for variable in ("v", "w", "z"):
                expected = getattr(hand, variable)
                assert toml_schema.state[variable] == expected
                assert json_schema.state[variable] == expected

        assert spike_count == 17
        assert rearm_count == 17
