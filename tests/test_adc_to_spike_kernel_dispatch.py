# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDispatch from former test_adc_to_spike_kernel.py

"""Focused suite: TestDispatch from former test_adc_to_spike_kernel.py."""

from __future__ import annotations

from tests.adc_to_spike_kernel_support import *  # noqa: F403


class TestDispatch:
    """Backend dispatch and availability."""

    @staticmethod
    def _samples() -> nptyping.NDArray[np.int64]:
        rng = np.random.default_rng(5)
        return rng.integers(0, 1 << 16, size=8 * 20, dtype=np.int64)

    def test_explicit_python_backend(self) -> None:
        """Explicit Python dispatch returns the bit-true floor result."""
        samples = self._samples()
        ref = adc_to_spike_windows_q(samples)
        out = adc_to_spike_windows(samples, backend="python")
        npt.assert_array_equal(out.window_values_q, ref.window_values_q)

    def test_auto_backend_matches_python(self) -> None:
        """Automatic dispatch preserves all public result arrays exactly."""
        samples = self._samples()
        ref = adc_to_spike_windows_q(samples)
        out = adc_to_spike_windows(samples, backend="auto")
        npt.assert_array_equal(out.window_values_q, ref.window_values_q)
        npt.assert_array_equal(out.spike_counts, ref.spike_counts)
        npt.assert_array_equal(out.polarities, ref.polarities)

    def test_unknown_backend_rejected(self) -> None:
        """Unknown backend names fail closed instead of silently using Python."""
        with pytest.raises(ValueError, match="unknown backend"):
            adc_to_spike_windows([1 << 15] * 8, backend="cuda")

    def test_available_backends_reports_python(self) -> None:
        """Backend probing always reports the Python floor and every known accelerator."""
        status = available_backends()
        assert status["python"] is True
        assert set(status) == set(kernel.FASTEST_FIRST_BACKENDS)

    def test_auto_falls_back_to_python(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Automatic dispatch falls back to Python when accelerators are unavailable."""

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
        """Backend probing reports unavailable accelerators without failing the probe."""

        def _boom(*_args: object, **_kwargs: object) -> ADCSpikeWindowResult:
            raise OSError("library missing")

        patched = dict(kernel._BACKEND_DISPATCH)
        patched["julia"] = _boom
        monkeypatch.setattr(kernel, "_BACKEND_DISPATCH", patched)
        status = available_backends()
        assert status["julia"] is False
        assert status["python"] is True
