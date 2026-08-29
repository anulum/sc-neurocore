# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Ibarz-Tanaka 2007 Q16.16 co-simulation

"""Three-way source-map and generated-RTL trajectory contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from sc_neurocore.neurons.models.ibarz_tanaka_map import IbarzTanakaMapNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import HAS_IVERILOG, _ibarz_tanaka_verilog_q1616_trace


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
def test_q1616_source_trajectory_and_reset_events() -> None:
    """Q16.16 preserves every source branch and the reset event vector."""
    schema_dir = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
    branch_counts = {"constant": 0, "parabolic": 0, "plateau": 0, "reset": 0}
    protocol_events: list[int] = []

    # I=-0.5 reaches the constant branch immediately; I=0.2 exercises the
    # parabolic, plateau, and source reset branches in one non-trivial train.
    for current, n_steps in ((-0.5, 4), (0.2, 30)):
        hand = IbarzTanakaMapNeuron()
        toml_schema = UniversalNeuron.from_schema(schema_dir / "ibarz_tanaka_map.toml")
        json_schema = UniversalNeuron.from_schema(schema_dir / "ibarz_tanaka_map.json")
        hand_trace: list[tuple[int, float, float]] = []
        for _step in range(n_steps):
            lower = -1.0 - hand.alpha / 2.0
            upper = 1.0 + current + hand.u
            if hand.v < lower:
                branch_counts["constant"] += 1
            elif hand.v <= 0.0:
                branch_counts["parabolic"] += 1
            elif hand.v < upper:
                branch_counts["plateau"] += 1
            else:
                branch_counts["reset"] += 1
            hand_event = hand.step(current)
            assert int(bool(toml_schema.step(I=current))) == hand_event
            assert int(bool(json_schema.step(I=current))) == hand_event
            assert toml_schema.state == {"v": hand.v, "u": hand.u}
            assert json_schema.state == {"v": hand.v, "u": hand.u}
            hand_trace.append((hand_event, hand.v, hand.u))

        rtl_trace = _ibarz_tanaka_verilog_q1616_trace(n_steps, current)
        assert [row[0] for row in hand_trace] == [row[0] for row in rtl_trace]
        protocol_events.append(sum(row[0] for row in rtl_trace))
        for (_event, expected_v, expected_u), (_rtl_event, rtl_v, rtl_u) in zip(
            hand_trace, rtl_trace, strict=True
        ):
            assert rtl_v == pytest.approx(expected_v, abs=0.003)
            assert rtl_u == pytest.approx(expected_u, abs=0.0001)

    assert branch_counts == {"constant": 3, "parabolic": 24, "plateau": 4, "reset": 3}
    assert protocol_events == [0, 3]
