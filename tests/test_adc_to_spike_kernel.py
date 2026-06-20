# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ADC-to-spike window encoder — unit + dispatch tests

"""Algorithm, validation and dispatch tests for the ADC-to-spike encoder.

Cross-language bit-exact parity and golden-reference parity are exercised in
``tests/test_adc_to_spike_kernel_parity.py``.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from sc_neurocore.sensors import adc_to_spike_kernel as kernel
from sc_neurocore.sensors.adc_to_spike_kernel import (
    ADCSpikeWindowConfig,
    ADCSpikeWindowResult,
    adc_to_spike_windows,
    adc_to_spike_windows_q,
    available_backends,
    quantise_adc,
)


class TestQuantiseAdc:
    """Sample centring and Q-format conversion across the three width regimes."""

    def test_equal_width_signed_identity(self) -> None:
        cfg = ADCSpikeWindowConfig(adc_width=16, q_int=8, q_frac=8, signed_input=True)
        assert quantise_adc(0, cfg) == 0
        assert quantise_adc(1, cfg) == 1
        assert quantise_adc((1 << 16) - 1, cfg) == -1  # two's-complement -1

    def test_offset_binary_centres_mid_scale(self) -> None:
        cfg = ADCSpikeWindowConfig(adc_width=16, q_int=8, q_frac=8, signed_input=False)
        assert quantise_adc(1 << 15, cfg) == 0  # mid-scale -> zero
        assert quantise_adc(0, cfg) == -(1 << 15)  # bottom -> most negative

    def test_up_shift_when_q_total_exceeds_adc_width(self) -> None:
        cfg = ADCSpikeWindowConfig(adc_width=12, q_int=8, q_frac=8, signed_input=True)
        # centred 1 -> 1 << (16 - 12) = 16.
        assert quantise_adc(1, cfg) == 16

    def test_round_down_signed_both_directions(self) -> None:
        cfg = ADCSpikeWindowConfig(adc_width=20, q_int=8, q_frac=8, signed_input=True)
        # positive: (16 + 8) >> 4 = 1; negative centred -16 -> (-16 - 8) >> 4 = -2.
        assert quantise_adc(16, cfg) == 1
        assert quantise_adc((1 << 20) - 16, cfg) == -2

    def test_saturates_to_q_bounds(self) -> None:
        cfg = ADCSpikeWindowConfig(adc_width=16, q_int=4, q_frac=4, signed_input=True)
        # Large positive sample round-down still exceeds Q4.4 max -> clamp to q_max.
        assert quantise_adc((1 << 15) - 1, cfg) == cfg.q_max
        assert quantise_adc(1 << 15, cfg) == cfg.q_min


class TestConfig:
    """Config invariants and validation."""

    def test_q_bounds(self) -> None:
        cfg = ADCSpikeWindowConfig(q_int=8, q_frac=8)
        assert cfg.q_total == 16
        assert cfg.q_min == -32768
        assert cfg.q_max == 32767

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"adc_width": 1}, "adc_width"),
            ({"q_int": 0}, "Q-format"),
            ({"q_frac": -1}, "Q-format"),
            ({"decimation": 0}, "decimation"),
            ({"threshold_q": 0}, "threshold_q"),
        ],
    )
    def test_validate_rejects(self, overrides: dict[str, int], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            ADCSpikeWindowConfig(**overrides).validate()


class TestWindows:
    """Per-window encode."""

    def test_offset_binary_zero_window_is_full_negative(self) -> None:
        cfg = ADCSpikeWindowConfig(signed_input=False, decimation=8, threshold_q=256)
        result = adc_to_spike_windows_q([0] * 8, cfg)
        assert isinstance(result, ADCSpikeWindowResult)
        assert result.window_values_q[0] == -32768
        assert result.spike_counts[0] == 128
        assert bool(result.polarities[0]) is True

    def test_average_negative_total_truncates_toward_zero(self) -> None:
        cfg = ADCSpikeWindowConfig(adc_width=16, q_int=8, q_frac=8, decimation=8, threshold_q=1)
        # Eight two's-complement -1 samples -> total -8 -> adjusted -12 -> -12//8 trunc = -1.
        samples = [(1 << 16) - 1] * 8
        result = adc_to_spike_windows_q(samples, cfg)
        assert result.window_values_q[0] == -1
        assert result.spike_counts[0] == 1
        assert bool(result.polarities[0]) is True

    def test_multiple_windows(self) -> None:
        cfg = ADCSpikeWindowConfig(decimation=4)
        result = adc_to_spike_windows_q(list(range(12)), cfg)
        assert result.window_values_q.shape == (3,)
        assert result.spike_counts.shape == (3,)

    def test_default_config_used_when_none(self) -> None:
        result = adc_to_spike_windows_q([1 << 15] * 8)
        assert result.window_values_q.shape == (1,)

    def test_too_few_samples_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least decimation"):
            adc_to_spike_windows_q([1, 2, 3], ADCSpikeWindowConfig(decimation=8))


class TestDispatch:
    """Backend dispatch and availability."""

    @staticmethod
    def _samples() -> np.ndarray:
        rng = np.random.default_rng(5)
        return rng.integers(0, 1 << 16, size=8 * 20, dtype=np.int64)

    def test_explicit_python_backend(self) -> None:
        samples = self._samples()
        ref = adc_to_spike_windows_q(samples)
        out = adc_to_spike_windows(samples, backend="python")
        npt.assert_array_equal(out.window_values_q, ref.window_values_q)

    def test_auto_backend_matches_python(self) -> None:
        samples = self._samples()
        ref = adc_to_spike_windows_q(samples)
        out = adc_to_spike_windows(samples, backend="auto")
        npt.assert_array_equal(out.window_values_q, ref.window_values_q)
        npt.assert_array_equal(out.spike_counts, ref.spike_counts)
        npt.assert_array_equal(out.polarities, ref.polarities)

    def test_unknown_backend_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown backend"):
            adc_to_spike_windows([1 << 15] * 8, backend="cuda")

    def test_available_backends_reports_python(self) -> None:
        status = available_backends()
        assert status["python"] is True
        assert set(status) == set(kernel.FASTEST_FIRST_BACKENDS)

    def test_auto_falls_back_to_python(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(*_args: object, **_kwargs: object) -> ADCSpikeWindowResult:
            raise ImportError("forced unavailable")

        patched = dict(kernel._BACKEND_DISPATCH)
        for name in ("rust", "mojo", "julia", "go"):
            patched[name] = _boom
        monkeypatch.setattr(kernel, "_BACKEND_DISPATCH", patched)
        samples = self._samples()
        ref = adc_to_spike_windows_q(samples)
        out = adc_to_spike_windows(samples, backend="auto")
        npt.assert_array_equal(out.window_values_q, ref.window_values_q)

    def test_available_backends_marks_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(*_args: object, **_kwargs: object) -> ADCSpikeWindowResult:
            raise OSError("library missing")

        patched = dict(kernel._BACKEND_DISPATCH)
        patched["julia"] = _boom
        monkeypatch.setattr(kernel, "_BACKEND_DISPATCH", patched)
        status = available_backends()
        assert status["julia"] is False
        assert status["python"] is True
