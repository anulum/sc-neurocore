# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQuantiseAdc from former test_adc_to_spike_kernel.py

"""Focused suite: TestQuantiseAdc from former test_adc_to_spike_kernel.py."""

from __future__ import annotations

from tests.adc_to_spike_kernel_support import *  # noqa: F403

class TestQuantiseAdc:
    """Sample centring and Q-format conversion across the three width regimes."""

    def test_equal_width_signed_identity(self) -> None:
        """Signed samples with matching ADC/Q widths preserve two's-complement codes."""
        cfg = ADCSpikeWindowConfig(adc_width=16, q_int=8, q_frac=8, signed_input=True)
        assert quantise_adc(0, cfg) == 0
        assert quantise_adc(1, cfg) == 1
        assert quantise_adc((1 << 16) - 1, cfg) == -1  # two's-complement -1

    def test_offset_binary_centres_mid_scale(self) -> None:
        """Offset-binary input recentres mid-scale to zero before Q conversion."""
        cfg = ADCSpikeWindowConfig(adc_width=16, q_int=8, q_frac=8, signed_input=False)
        assert quantise_adc(1 << 15, cfg) == 0  # mid-scale -> zero
        assert quantise_adc(0, cfg) == -(1 << 15)  # bottom -> most negative

    def test_up_shift_when_q_total_exceeds_adc_width(self) -> None:
        """Narrow ADC samples are left-shifted into the wider Q-format."""
        cfg = ADCSpikeWindowConfig(adc_width=12, q_int=8, q_frac=8, signed_input=True)
        # centred 1 -> 1 << (16 - 12) = 16.
        assert quantise_adc(1, cfg) == 16

    def test_round_down_signed_both_directions(self) -> None:
        """Wide ADC samples use sign-aware half-offset rounding before down-shift."""
        cfg = ADCSpikeWindowConfig(adc_width=20, q_int=8, q_frac=8, signed_input=True)
        # positive: (16 + 8) >> 4 = 1; negative centred -16 -> (-16 - 8) >> 4 = -2.
        assert quantise_adc(16, cfg) == 1
        assert quantise_adc((1 << 20) - 16, cfg) == -2

    def test_saturates_to_q_bounds(self) -> None:
        """Quantisation clamps samples that exceed the configured Q-format range."""
        cfg = ADCSpikeWindowConfig(adc_width=16, q_int=4, q_frac=4, signed_input=True)
        # Large positive sample round-down still exceeds Q4.4 max -> clamp to q_max.
        assert quantise_adc((1 << 15) - 1, cfg) == cfg.q_max
        assert quantise_adc(1 << 15, cfg) == cfg.q_min
