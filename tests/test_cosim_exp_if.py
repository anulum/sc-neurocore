# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ExpIF Python-to-Verilog fidelity contracts

"""Source equation, paired-schema, and Q32.32 RTL parity for ExpIF."""

from __future__ import annotations

from pathlib import Path

import pytest

from sc_neurocore.neurons.models.expif import ExpIFNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import HAS_IVERILOG, _python_spike_count, verilog_spike_count_method

_Q3232_GOLDENS = ((0.0, 0), (5.0, 0), (10.0, 1), (20.0, 2), (50.0, 5), (100.0, 9))


def test_expif_schema_formats_match_the_hand_rk4_sequence() -> None:
    """TOML and JSON preserve operation order over a varied driven sequence."""
    schema_dir = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
    hand = ExpIFNeuron()
    toml_schema = UniversalNeuron.from_schema(schema_dir / "exp_if.toml")
    json_schema = UniversalNeuron.from_schema(schema_dir / "exp_if.json")
    currents = (0.0, 5.0, 10.0, 20.0, 50.0, 100.0, 20.0, 5.0) * 125
    spikes = 0
    max_error = 0.0

    for current in currents:
        hand_spike = hand.step(current)
        spikes += hand_spike
        assert int(bool(toml_schema.step(I=current))) == hand_spike
        assert int(bool(json_schema.step(I=current))) == hand_spike
        max_error = max(
            max_error,
            abs(toml_schema.state["v"] - hand.v),
            abs(json_schema.state["v"] - hand.v),
        )

    assert spikes > 0
    assert max_error <= 2.0e-10


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
@pytest.mark.parametrize(
    ("current", "expected_spikes"),
    _Q3232_GOLDENS,
    ids=[f"I={current:g}" for current, _expected in _Q3232_GOLDENS],
)
def test_expif_q3232_spike_parity(current: float, expected_spikes: int) -> None:
    """Match hand, schema, and emitted Q32.32 events over 1,000 RK4 steps.

    Q16.16 cannot represent the steep pre-cutoff exponential product without
    losing the event train. Q32.32 retains enough integer and fractional range;
    the six enrolled points span silence, onset, and sustained firing.
    """
    n_steps = 1_000
    hand = ExpIFNeuron()
    hand_spikes = sum(hand.step(current) for _ in range(n_steps))
    schema_spikes = _python_spike_count("exp_if", n_steps, current)
    verilog_spikes = verilog_spike_count_method("exp_if", n_steps, current, 64, 32, "rk4")
    assert hand_spikes == schema_spikes == verilog_spikes == expected_spikes
