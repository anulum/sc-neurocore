# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestStochasticSTDPSynapse from former test_synapses_stdp.py

"""Focused suite: TestStochasticSTDPSynapse from former test_synapses_stdp.py."""

from __future__ import annotations

from tests.synapses_stdp_support import *  # noqa: F403


class TestStochasticSTDPSynapse:
    def _make(self, w=0.5, lr=0.01, seed=42):
        return StochasticSTDPSynapse(
            w_min=0.0,
            w_max=1.0,
            length=256,
            w=w,
            learning_rate=lr,
            window_size=5,
            seed=seed,
        )

    def test_construction(self):
        syn = self._make()
        assert syn.w == pytest.approx(0.5)
        assert syn._pre_trace.shape == (5,)
        assert np.all(syn._pre_trace == 0)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("learning_rate", -0.1),
            ("learning_rate", 1.1),
            ("learning_rate", float("nan")),
            ("window_size", 0),
            ("window_size", -1),
            ("window_size", True),
            ("ltd_ratio", -0.1),
            ("ltd_ratio", float("inf")),
        ],
    )
    def test_invalid_stdp_parameters_fail_closed(self, field, value):
        kwargs = {
            "w_min": 0.0,
            "w_max": 1.0,
            "length": 256,
            "w": 0.5,
            "learning_rate": 0.01,
            "window_size": 5,
            "seed": 42,
        }
        kwargs[field] = value
        with pytest.raises(ValueError, match=field):
            StochasticSTDPSynapse(**kwargs)

    @pytest.mark.parametrize(
        ("pre_bit", "post_bit"),
        [
            (2, 0),
            (-1, 1),
            (1, 2),
            (True, 0),
            (1, False),
        ],
    )
    def test_invalid_stdp_step_bits_fail_closed(self, pre_bit, post_bit):
        syn = self._make()
        with pytest.raises(ValueError, match="bit"):
            syn.process_step(pre_bit=pre_bit, post_bit=post_bit)

    def test_process_step_returns_binary(self):
        syn = self._make()
        for _ in range(100):
            out = syn.process_step(pre_bit=1, post_bit=0)
            assert out in (0, 1)

    def test_output_is_and_of_pre_and_weight(self):
        """Output is pre_bit AND weight_bit; when pre=0, output must be 0."""
        syn = self._make()
        for _ in range(50):
            assert syn.process_step(pre_bit=0, post_bit=1) == 0

    def test_ltp_increases_weight(self):
        """Sustained pre=1, post=1 should increase weight over many steps."""
        syn = self._make(w=0.3, lr=0.05, seed=0)
        initial_w = syn.w
        for _ in range(500):
            syn.process_step(pre_bit=1, post_bit=1)
        assert syn.w > initial_w

    def test_ltd_decreases_weight(self):
        """Sustained pre=1, post=0 should decrease weight over many steps."""
        syn = self._make(w=0.7, lr=0.05, seed=0)
        initial_w = syn.w
        for _ in range(500):
            syn.process_step(pre_bit=1, post_bit=0)
        assert syn.w < initial_w

    def test_weight_stays_in_bounds(self):
        """Weight should never exceed [w_min, w_max] regardless of input."""
        syn = self._make(w=0.99, lr=0.1, seed=0)
        for _ in range(1000):
            syn.process_step(pre_bit=1, post_bit=1)
        assert syn.w <= syn.w_max

        syn2 = self._make(w=0.01, lr=0.1, seed=0)
        for _ in range(1000):
            syn2.process_step(pre_bit=1, post_bit=0)
        assert syn2.w >= syn2.w_min

    def test_potentiate_directly(self):
        syn = self._make(w=0.5)
        syn._potentiate()
        assert syn.w > 0.5

    def test_depress_directly(self):
        syn = self._make(w=0.5)
        syn._depress()
        assert syn.w < 0.5

    def test_pre_trace_shifts(self):
        """Pre-trace buffer should shift bits in correctly."""
        syn = self._make()
        syn.process_step(pre_bit=1, post_bit=0)
        assert syn._pre_trace[0] == 1
        syn.process_step(pre_bit=0, post_bit=0)
        assert syn._pre_trace[0] == 0
        assert syn._pre_trace[1] == 1  # shifted
