# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — retained upward-crossing Rulkov Q16.16 co-simulation

"""Hand/schema/RTL parity for the retained Rulkov event convention."""

from __future__ import annotations

from pathlib import Path

import pytest

from sc_neurocore.neurons.models.sc_upward_crossing_rulkov_map import (
    SCUpwardCrossingRulkovMapNeuron,
)
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_reference_sc_upward_crossing_rulkov_map import (
    sc_upward_crossing_rulkov_q1616_trace,
)
from tests.cosim_runtime import HAS_IVERILOG


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
@pytest.mark.parametrize("current", (0.0, 0.5, 1.5))
def test_sc_upward_crossing_rulkov_q1616_short_window(current: float) -> None:
    """Require exact events and bounded states across hand, schema, and RTL."""
    n_steps = 30
    schema_root = Path(__file__).parents[1] / "src/sc_neurocore/neurons/model_schemas"
    hand = SCUpwardCrossingRulkovMapNeuron()
    toml_schema = UniversalNeuron.from_schema(schema_root / "sc_upward_crossing_rulkov_map.toml")
    json_schema = UniversalNeuron.from_schema(schema_root / "sc_upward_crossing_rulkov_map.json")
    expected: list[tuple[int, float, float]] = []
    for _ in range(n_steps):
        event = hand.step(current)
        assert bool(toml_schema.step(I=current)) is bool(event)
        assert bool(json_schema.step(I=current)) is bool(event)
        assert toml_schema.state == {"x": hand.x, "y": hand.y}
        assert json_schema.state == {"x": hand.x, "y": hand.y}
        expected.append((event, hand.x, hand.y))
    rtl = sc_upward_crossing_rulkov_q1616_trace(n_steps, current)
    assert [row[0] for row in rtl] == [row[0] for row in expected]
    for (_event, x_expected, y_expected), (_rtl_event, x_rtl, y_rtl) in zip(
        expected, rtl, strict=True
    ):
        assert x_rtl == pytest.approx(x_expected, abs=0.006)
        assert y_rtl == pytest.approx(y_expected, abs=0.001)
