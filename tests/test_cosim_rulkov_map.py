# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rulkov map co-simulation contracts

"""Rulkov map Q16.16 parity contracts."""

from __future__ import annotations

from pathlib import Path

from sc_neurocore.neurons.models.rulkov_map import RulkovMapNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
import pytest

from tests.cosim_support import (
    HAS_IVERILOG,
    _rulkov_map_verilog_q1616_trace,
)


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestQ1616Precision:
    """Q16.16 precision mode: 16 integer + 16 fractional bits (32-bit).

    Q16.16 combines Q8.8's wide integer range [-32768, +32767] with
    1/65536 ≈ 0.000015 resolution. This is the "gold standard" for
    hardware neuron fidelity, suitable for all model dynamics.
    """

    @pytest.mark.parametrize(
        ("current", "expected_events", "x_tolerance", "y_tolerance"),
        (
            (0.0, 0, 0.0011, 0.0003),
            (0.5, 7, 0.0055, 0.0006),
            (1.5, 10, 0.001, 0.0008),
        ),
    )
    def test_rulkov_map_q1616_short_window_trajectory(
        self,
        current: float,
        expected_events: int,
        x_tolerance: float,
        y_tolerance: float,
    ) -> None:
        """Rulkov has class-correct three-way short-window trajectory parity.

        The maintained hand model and both schema formats execute the published
        simultaneous fast/slow map with source rightmost-branch event detection.
        The three enrolled drives cover quiescent, moderate, and rapid map
        regimes. At ``I=1.5`` the window visits every branch ten times.
        Hand/TOML/JSON decisions and states must be exact; emitted Q16.16 RTL
        must reproduce each complete event vector within measured state bounds.
        No long-window fixed-point trajectory identity is claimed.
        """
        n_steps = 30
        schema_dir = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
        hand = RulkovMapNeuron()
        toml_schema = UniversalNeuron.from_schema(schema_dir / "rulkov_map.toml")
        json_schema = UniversalNeuron.from_schema(schema_dir / "rulkov_map.json")
        hand_trace: list[tuple[int, float, float]] = []
        branch_counts = {"rational": 0, "plateau": 0, "reset": 0}

        for _step in range(n_steps):
            boundary = hand.alpha + hand.y + current
            if hand.x <= 0.0:
                branch_counts["rational"] += 1
            elif hand.x < boundary:
                branch_counts["plateau"] += 1
            else:
                branch_counts["reset"] += 1
            hand_spike = hand.step(current)
            assert int(bool(toml_schema.step(I=current))) == hand_spike
            assert int(bool(json_schema.step(I=current))) == hand_spike
            assert toml_schema.state == {"x": hand.x, "y": hand.y}
            assert json_schema.state == {"x": hand.x, "y": hand.y}
            hand_trace.append((hand_spike, hand.x, hand.y))

        rtl_trace = _rulkov_map_verilog_q1616_trace(n_steps, current)
        if current == 1.5:
            assert branch_counts == {"rational": 10, "plateau": 10, "reset": 10}
        assert [row[0] for row in hand_trace] == [row[0] for row in rtl_trace]
        assert sum(row[0] for row in rtl_trace) == expected_events
        for (_spike, expected_x, expected_y), (_rtl_spike, rtl_x, rtl_y) in zip(
            hand_trace, rtl_trace, strict=True
        ):
            assert rtl_x == pytest.approx(expected_x, abs=x_tolerance)
            assert rtl_y == pytest.approx(expected_y, abs=y_tolerance)
