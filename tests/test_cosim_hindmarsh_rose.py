# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hindmarsh-Rose co-simulation contracts

"""Hindmarsh-Rose schema, hand-model, and Q16.16 RTL parity contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from sc_neurocore.neurons.models.hindmarsh_rose import HindmarshRoseNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import (
    HAS_IVERILOG,
    _hindmarsh_rose_hand_spike_count,
    _python_spike_count,
    _verilog_spike_count_q1616,
)


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


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestQ1616Precision:
    """Q16.16 Hindmarsh-Rose co-simulation fidelity."""

    @pytest.mark.parametrize(
        ("current", "expected_spikes"),
        ((0.0, 0), (2.0, 0), (3.0, 26), (4.0, 40), (5.0, 52)),
        ids=("I=0", "I=2", "I=3", "I=4", "I=5"),
    )
    def test_hindmarsh_rose_q1616_parity(self, current: float, expected_spikes: int) -> None:
        """The hand model, schema, and Q16.16 RTL agree over 2,000 RK4 steps.

        The enrolled set spans two silent points and three bursting rates. The schema
        mirrors the maintained simultaneous three-state RK4 flow, rising-edge
        ``x >= x_threshold`` observation, and no-reset semantics. The 2,000-step
        horizon is deliberate: longer chaotic trajectories are separately classified
        below instead of being presented as indefinite fixed-point identity.
        """
        n_steps = 2000
        hand_spikes = _hindmarsh_rose_hand_spike_count(n_steps, current)
        schema_spikes = _python_spike_count("hindmarsh_rose", n_steps, current)
        rtl_spikes = _verilog_spike_count_q1616("hindmarsh_rose", n_steps, current)

        assert hand_spikes == schema_spikes == rtl_spikes == expected_spikes, (
            f"Hindmarsh-Rose three-way mismatch at I={current}: hand={hand_spikes}, "
            f"schema={schema_spikes}, verilog={rtl_spikes}"
        )

    @pytest.mark.parametrize(
        ("current", "expected_float", "expected_rtl"),
        ((2.0, 9, 10), (3.0, 48, 49), (4.0, 85, 86), (5.0, 114, 115)),
        ids=("I=2", "I=3", "I=4", "I=5"),
    )
    def test_hindmarsh_rose_q1616_long_window_boundary(
        self, current: float, expected_float: int, expected_rtl: int
    ) -> None:
        """The 5,000-step chaotic boundary is an explicit one-crossing exclusion."""
        n_steps = 5000
        hand_spikes = _hindmarsh_rose_hand_spike_count(n_steps, current)
        schema_spikes = _python_spike_count("hindmarsh_rose", n_steps, current)
        rtl_spikes = _verilog_spike_count_q1616("hindmarsh_rose", n_steps, current)

        assert hand_spikes == schema_spikes == expected_float
        assert rtl_spikes == expected_rtl == expected_float + 1
