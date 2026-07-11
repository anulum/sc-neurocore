# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mihalas-Niebur co-simulation contracts

"""Mihalas-Niebur schema, hand-model, and Q16.16 RTL parity contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from sc_neurocore.neurons.models.mihalas_niebur import MihalasNieburNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import (
    HAS_IVERILOG,
    _MIHALAS_NIEBUR_PARAMS,
    _mihalas_niebur_hand_spike_count,
    _python_spike_count,
    _verilog_spike_count_q1616,
)


class TestTierBModelCosim:
    """WC-A5 Tier-B Mihalas-Niebur enrolment."""

    def test_mihalas_niebur_schema_matches_hand_rk4_sequence(self) -> None:
        """Both schemas mirror the Mihalas-Niebur RK4 flow and adaptive reset.

        The paired TOML/JSON ``mihalas_niebur`` schemas are the ``method="rk4"``
        discretisation of the generalised integrate-and-fire neuron
        (``MihalasNieburNeuron``, Mihalaş & Niebur 2009). The 1,600-step varied drive
        exercises the four coupled states, every RK4 stage, silence, tonic firing, and
        168 candidate-first resets. Both schema formats must reproduce every hand-model
        event and post-step state exactly.
        """
        schema_dir = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
        hand = MihalasNieburNeuron(dt=1.0, **_MIHALAS_NIEBUR_PARAMS)
        toml_schema = UniversalNeuron.from_schema(schema_dir / "mihalas_niebur.toml")
        json_schema = UniversalNeuron.from_schema(schema_dir / "mihalas_niebur.json")
        currents = (0.0, 3.0, 5.0, 2.0, 4.0, 0.0, 6.0, 3.5) * 200
        spike_count = 0

        for current in currents:
            hand_spike = hand.step(current)
            spike_count += hand_spike
            assert int(bool(toml_schema.step(I=current))) == hand_spike
            assert int(bool(json_schema.step(I=current))) == hand_spike
            for variable in ("v", "theta", "i1", "i2"):
                expected = getattr(hand, variable)
                assert toml_schema.state[variable] == expected
                assert json_schema.state[variable] == expected

        assert spike_count == 168

    @pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
    def test_mihalas_niebur_q1616_legacy_window_is_exact(self) -> None:
        """The corrected candidate-reset RTL exactly matches the former 300-step window.

        At ``I=3.0`` the maintained hand model, schema runner, and emitted Q16.16 RTL now
        produce the same 36-spike partial train. This guards the post-candidate reset/output
        semantics that replaced the stale 36/36/35 evidence.
        """
        hand_spikes = _mihalas_niebur_hand_spike_count(300, 3.0)
        schema_spikes = _python_spike_count("mihalas_niebur", 300, 3.0)
        verilog_spikes = _verilog_spike_count_q1616("mihalas_niebur", 300, 3.0)

        assert hand_spikes == schema_spikes == verilog_spikes == 36

    @pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
    @pytest.mark.parametrize(
        ("current", "expected_spikes"),
        (
            (0.0, 0),
            (0.5, 0),
            (1.0, 0),
            (1.5, 31),
            (2.0, 60),
            (2.5, 87),
            (3.5, 131),
            (4.0, 157),
            (5.0, 207),
            (6.0, 256),
        ),
        ids=(
            "rest",
            "subthreshold-low",
            "subthreshold-high",
            "onset",
            "low-train",
            "medium-train",
            "above-boundary",
            "tonic",
            "high-drive",
            "strong-drive",
        ),
    )
    def test_mihalas_niebur_q1616_exact_operating_set(
        self, current: float, expected_spikes: int
    ) -> None:
        """Mihalas-Niebur has exact Q16.16 parity at ten enrolled currents.

        The set spans three silent regimes and seven partial trains over 1,000 RK4 steps.
        Hand-model and schema equality anchors the float64 formulation; equality with the
        emitted RTL proves fixed-point spike-count parity on both sides of the isolated
        ``I=3.0`` crossing boundary.
        """
        n_steps = 1000
        hand_spikes = _mihalas_niebur_hand_spike_count(n_steps, current)
        schema_spikes = _python_spike_count("mihalas_niebur", n_steps, current)
        verilog_spikes = _verilog_spike_count_q1616("mihalas_niebur", n_steps, current)

        assert hand_spikes == schema_spikes == verilog_spikes == expected_spikes, (
            f"Mihalas-Niebur exact Q16.16 mismatch at I={current}: "
            f"hand={hand_spikes}, schema={schema_spikes}, verilog={verilog_spikes}"
        )

    @pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
    def test_mihalas_niebur_q1616_declares_i3_boundary(self) -> None:
        """The 1,000-step ``I=3.0`` crossing boundary remains explicit and exact.

        Q16.16 rounding advances one marginal adaptive-threshold crossing: the hand model
        and schema runner produce 111 spikes while RTL produces 112. Pinning the complete
        triplet prevents either hiding the boundary behind a loose tolerance or promoting
        the operating point to exact parity.
        """
        n_steps = 1000
        hand_spikes = _mihalas_niebur_hand_spike_count(n_steps, 3.0)
        schema_spikes = _python_spike_count("mihalas_niebur", n_steps, 3.0)
        verilog_spikes = _verilog_spike_count_q1616("mihalas_niebur", n_steps, 3.0)

        assert (hand_spikes, schema_spikes, verilog_spikes) == (111, 111, 112)
