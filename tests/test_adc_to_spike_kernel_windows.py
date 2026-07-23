# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWindows from former test_adc_to_spike_kernel.py

"""Focused suite: TestWindows from former test_adc_to_spike_kernel.py."""

from __future__ import annotations

from tests.adc_to_spike_kernel_support import *  # noqa: F403

class TestWindows:
    """Per-window encode."""

    def test_offset_binary_zero_window_is_full_negative(self) -> None:
        """An all-zero offset-binary window emits the expected negative full-scale rate."""
        cfg = ADCSpikeWindowConfig(signed_input=False, decimation=8, threshold_q=256)
        result = adc_to_spike_windows_q([0] * 8, cfg)
        assert isinstance(result, ADCSpikeWindowResult)
        assert result.window_values_q[0] == -32768
        assert result.spike_counts[0] == 128
        assert bool(result.polarities[0]) is True

    def test_average_negative_total_truncates_toward_zero(self) -> None:
        """Negative window totals match the golden model's sign-aware truncation."""
        cfg = ADCSpikeWindowConfig(adc_width=16, q_int=8, q_frac=8, decimation=8, threshold_q=1)
        # Eight two's-complement -1 samples -> total -8 -> adjusted -12 -> -12//8 trunc = -1.
        samples = [(1 << 16) - 1] * 8
        result = adc_to_spike_windows_q(samples, cfg)
        assert result.window_values_q[0] == -1
        assert result.spike_counts[0] == 1
        assert bool(result.polarities[0]) is True

    def test_multiple_windows(self) -> None:
        """Only completed decimation windows are emitted in the result arrays."""
        cfg = ADCSpikeWindowConfig(decimation=4)
        result = adc_to_spike_windows_q(list(range(12)), cfg)
        assert result.window_values_q.shape == (3,)
        assert result.spike_counts.shape == (3,)

    def test_default_config_used_when_none(self) -> None:
        """The Q8.8 decimation-8 contract is the default encoder configuration."""
        result = adc_to_spike_windows_q([1 << 15] * 8)
        assert result.window_values_q.shape == (1,)

    def test_too_few_samples_rejected(self) -> None:
        """A stream shorter than one decimation window is rejected."""
        with pytest.raises(ValueError, match="at least decimation"):
            adc_to_spike_windows_q([1, 2, 3], ADCSpikeWindowConfig(decimation=8))
