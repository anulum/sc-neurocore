# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBitstreamCurrentSource from former test_sources_and_profiling.py

"""Focused suite: TestBitstreamCurrentSource from former test_sources_and_profiling.py."""

from __future__ import annotations

from tests.sources_and_profiling_support import *  # noqa: F403

class TestBitstreamCurrentSource:
    def test_construction(self):
        src = BitstreamCurrentSource(
            x_inputs=[0.5, 0.5],
            x_min=0.0,
            x_max=1.0,
            weight_values=[0.5, 0.5],
            w_min=0.0,
            w_max=1.0,
            length=256,
            seed=42,
        )
        assert src.n_inputs == 2
        assert src.pre_matrix.shape == (2, 256)
        assert src.post_matrix.shape == (2, 256)

    def test_input_weight_mismatch_raises(self):
        with pytest.raises(ValueError):
            BitstreamCurrentSource(
                x_inputs=[0.5, 0.5],
                x_min=0.0,
                x_max=1.0,
                weight_values=[0.5],
                w_min=0.0,
                w_max=1.0,
            )

    def test_step_returns_float_in_range(self):
        src = BitstreamCurrentSource(
            x_inputs=[0.8],
            x_min=0.0,
            x_max=1.0,
            weight_values=[0.6],
            w_min=0.0,
            w_max=1.0,
            length=64,
            y_min=0.0,
            y_max=1.0,
            seed=42,
        )
        for _ in range(64):
            I_t = src.step()
            assert 0.0 <= I_t <= 1.0

    def test_step_clamps_past_length(self):
        """After length steps, it should clamp at the last index."""
        src = BitstreamCurrentSource(
            x_inputs=[0.5],
            x_min=0.0,
            x_max=1.0,
            weight_values=[0.5],
            w_min=0.0,
            w_max=1.0,
            length=8,
            seed=42,
        )
        for _ in range(20):
            I_t = src.step()
        assert isinstance(I_t, float)

    def test_reset(self):
        src = BitstreamCurrentSource(
            x_inputs=[0.5],
            x_min=0.0,
            x_max=1.0,
            weight_values=[0.5],
            w_min=0.0,
            w_max=1.0,
            length=16,
            seed=42,
        )
        first_vals = [src.step() for _ in range(5)]
        src.reset()
        second_vals = [src.step() for _ in range(5)]
        assert first_vals == second_vals

    def test_full_current_estimate(self):
        src = BitstreamCurrentSource(
            x_inputs=[0.8, 0.6],
            x_min=0.0,
            x_max=1.0,
            weight_values=[0.5, 0.5],
            w_min=0.0,
            w_max=1.0,
            length=1024,
            y_min=0.0,
            y_max=0.1,
            seed=42,
        )
        est = src.full_current_estimate()
        assert isinstance(est, float)
        assert 0.0 <= est <= 0.1

    def test_high_weight_high_input_gives_more_current(self):
        """Higher weights and inputs should produce more current."""
        src_low = BitstreamCurrentSource(
            x_inputs=[0.2],
            x_min=0.0,
            x_max=1.0,
            weight_values=[0.2],
            w_min=0.0,
            w_max=1.0,
            length=1024,
            y_min=0.0,
            y_max=1.0,
            seed=42,
        )
        src_high = BitstreamCurrentSource(
            x_inputs=[0.9],
            x_min=0.0,
            x_max=1.0,
            weight_values=[0.9],
            w_min=0.0,
            w_max=1.0,
            length=1024,
            y_min=0.0,
            y_max=1.0,
            seed=42,
        )
        assert src_high.full_current_estimate() > src_low.full_current_estimate()
