# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRustMatchesC from former test_bit_true_cosim.py

"""Focused suite: TestRustMatchesC from former test_bit_true_cosim.py."""

from __future__ import annotations

from tests.bit_true_cosim_support import *  # noqa: F403

@pytest.mark.skipif(
    not (HAS_COSIM and HAS_RUST), reason="Icarus Verilog / gcc / rustc not all available"
)
class TestRustMatchesC:
    @pytest.mark.parametrize(
        "name,factory,current,steps,dw,frac", _CASES, ids=[c[0] for c in _CASES]
    )
    def test_rust_kernel_matches_c(
        self,
        name: str,
        factory: NeuronFactory,
        current: float,
        steps: int,
        dw: int,
        frac: int,
    ) -> None:
        neuron = factory()
        module = f"sc_{name}"
        i_q = _q_input(current, dw, frac)
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            ctrace = _c_trace(neuron, module, i_q, steps, dw, frac, tmp)
            rtrace = _rust_trace(neuron, module, i_q, steps, dw, frac, tmp)
        assert ctrace == rtrace, f"{name}: Rust kernel diverged from C kernel"

    def test_nearest_negative_half_tie_matches_rust_and_c(self) -> None:
        """Rust shares the signed half-tie contract proven against RTL by C."""
        neuron = _nearest_negative_half_tie_map()
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            c_trace = _c_trace(
                neuron,
                "sc_nearest_negative_half_c_rust",
                0,
                1,
                16,
                8,
                tmp,
                rounding="nearest",
            )
            rust = _rust_trace(
                neuron,
                "sc_nearest_negative_half_rust",
                0,
                1,
                16,
                8,
                tmp,
                rounding="nearest",
            )

        assert rust == c_trace == [["0", "-1"]]

    @pytest.mark.parametrize(
        ("current", "expected_events"),
        ((-0.5, 0), (1.0, 8)),
        ids=("negative-backward-wrap", "positive-upward-crossing"),
    )
    def test_ermentrout_hand_event_vector_matches_all_bittrue_targets(
        self,
        current: float,
        expected_events: int,
    ) -> None:
        """Both crossing signs agree across the hand model, RTL, C, and Rust."""
        steps = 240
        suffix = "negative" if current < 0.0 else "positive"
        module = f"sc_ek_all_targets_{suffix}"
        hand = ErmentroutKopellMapNeuron()
        hand_events = [str(hand.step(current)) for _ in range(steps)]
        neuron = _ermentrout_kopell_map()
        i_q = _q_input(current, 32, 16)

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            verilog = _verilog_trace(neuron, module, current, steps, 32, 16, tmp)
            c_trace = _c_trace(neuron, module, i_q, steps, 32, 16, tmp)
            rust = _rust_trace(neuron, module, i_q, steps, 32, 16, tmp)

        assert sum(int(event) for event in hand_events) == expected_events
        assert hand_events == [row[0] for row in verilog]
        assert hand_events == [row[0] for row in c_trace]
        assert hand_events == [row[0] for row in rust]
