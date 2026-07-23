# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBitTrueCosim from former test_bit_true_cosim.py

"""Focused suite: TestBitTrueCosim from former test_bit_true_cosim.py."""

from __future__ import annotations

from tests.bit_true_cosim_support import *  # noqa: F403

@pytest.mark.skipif(not HAS_COSIM, reason="Icarus Verilog / gcc not available")
class TestBitTrueCosim:
    @pytest.mark.parametrize(
        "name,factory,current,steps,dw,frac", _CASES, ids=[c[0] for c in _CASES]
    )
    def test_c_kernel_matches_rtl(
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
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            vtrace = _verilog_trace(neuron, module, current, steps, dw, frac, tmp)
            ctrace = _c_trace(neuron, module, _q_input(current, dw, frac), steps, dw, frac, tmp)
        assert vtrace, "Verilog produced no trace"
        assert len(vtrace) == len(ctrace) == steps
        assert vtrace == ctrace, f"{name}: bit-true kernel diverged from RTL"

    def test_fire_case_actually_resets(self) -> None:
        # Guard against a trivially-matching (monotone) trace: the fire case must
        # exercise the threshold / reset path — its output must drop at least once.
        neuron = _fire()
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            vtrace = _verilog_trace(neuron, "sc_fire_rst", 1.0, 30, 16, 8, tmp)
        v_series = [int(row[1]) for row in vtrace]
        drops = [b < a for a, b in zip(v_series, v_series[1:])]
        assert any(drops), "expected at least one reset (falling edge) in the trace"

    def test_sqrt_case_selects_the_exact_square_entry(self) -> None:
        """The first map step must lower ``sqrt(4)`` to the Q8.8 code for two."""
        with tempfile.TemporaryDirectory() as directory:
            trace = _verilog_trace(
                _sqrt_map(), "sc_sqrt_exact_square", 0.0, 1, 16, 8, Path(directory)
            )

        assert trace == [["0", "512"]]

    def test_nearest_negative_half_tie_rounds_away_from_zero(self) -> None:
        """RTL and C must map a negative half-LSB product to minus one LSB."""
        neuron = _nearest_negative_half_tie_map()
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            verilog = _verilog_trace(
                neuron,
                "sc_nearest_negative_half_rtl",
                0.0,
                1,
                16,
                8,
                tmp,
                rounding="nearest",
            )
            c_trace = _c_trace(
                neuron,
                "sc_nearest_negative_half_c",
                0,
                1,
                16,
                8,
                tmp,
                rounding="nearest",
            )

        assert verilog == c_trace == [["0", "-1"]]

    def test_ermentrout_kopell_cases_exercise_both_event_branches(self) -> None:
        """Positive drive crosses pi; a negative backward wrap stays event-silent."""
        neuron = _ermentrout_kopell_map()
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            negative = _verilog_trace(neuron, "sc_ek_negative", -0.5, 240, 32, 16, tmp)
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            positive = _verilog_trace(
                _ermentrout_kopell_map(), "sc_ek_positive", 1.0, 240, 32, 16, tmp
            )

        assert sum(int(row[0]) for row in negative) == 0
        assert int(negative[0][1]) > int(3.141592653589793 * (1 << 16))
        assert sum(int(row[0]) for row in positive) > 0
