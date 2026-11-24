# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDispatch from former test_dcls_tent_kernel.py

"""Focused suite: TestDispatch from former test_dcls_tent_kernel.py."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from sc_neurocore.scpn import (
    dcls_tent_kernel as kernel,
)
from sc_neurocore.scpn.dcls_tent_kernel import (
    available_backends,
    dcls_max_forward_batch,
    dcls_max_forward_batch_q88,
    DclsBatchResult,
)


class TestDispatch:
    """Backend dispatch and availability."""

    @staticmethod
    def _workload() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        rng = np.random.default_rng(3)
        n_channels, n_taps = 32, 16
        spikes = (rng.random(n_channels * n_taps) < 0.5).astype(np.uint8)
        weights = rng.integers(-2048, 2048, n_channels * n_taps, dtype=np.int16)
        centres = rng.integers(-128, 2048, n_channels, dtype=np.int16)
        sigmas = rng.integers(1, 2048, n_channels, dtype=np.int16)
        return spikes, weights, centres, sigmas, n_taps

    def test_explicit_python_backend(self) -> None:
        """The Python backend matches the reference."""
        spikes, weights, centres, sigmas, n_taps = self._workload()
        ref = dcls_max_forward_batch_q88(spikes, weights, centres, sigmas, n_taps)
        out = dcls_max_forward_batch(spikes, weights, centres, sigmas, n_taps, backend="python")
        npt.assert_array_equal(out.outputs_q88, ref.outputs_q88)

    def test_auto_backend_matches_python(self) -> None:
        """The automatic backend matches Python."""
        spikes, weights, centres, sigmas, n_taps = self._workload()
        ref = dcls_max_forward_batch_q88(spikes, weights, centres, sigmas, n_taps)
        out = dcls_max_forward_batch(spikes, weights, centres, sigmas, n_taps, backend="auto")
        npt.assert_array_equal(out.outputs_q88, ref.outputs_q88)
        npt.assert_array_equal(out.accumulators_q16_16, ref.accumulators_q16_16)

    def test_unknown_backend_rejected(self) -> None:
        """An unknown backend is rejected."""
        with pytest.raises(ValueError, match="unknown backend"):
            dcls_max_forward_batch([1], [256], [256], [512], 1, backend="cuda")

    def test_available_backends_reports_python(self) -> None:
        """Python is always reported available."""
        status = available_backends()
        assert status["python"] is True
        assert set(status) == set(kernel.FASTEST_FIRST_BACKENDS)

    def test_auto_falls_back_to_python(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without a native backend the call falls back to Python."""

        # Force every accelerator to fail so the auto loop exhausts and lands on
        # the Python floor.
        def _boom(*_args: object, **_kwargs: object) -> DclsBatchResult:
            raise ImportError("forced unavailable")

        patched = dict(kernel._BACKEND_DISPATCH)
        for name in ("rust", "mojo", "julia", "go"):
            patched[name] = _boom
        monkeypatch.setattr(kernel, "_BACKEND_DISPATCH", patched)
        spikes, weights, centres, sigmas, n_taps = self._workload()
        ref = dcls_max_forward_batch_q88(spikes, weights, centres, sigmas, n_taps)
        out = dcls_max_forward_batch(spikes, weights, centres, sigmas, n_taps, backend="auto")
        npt.assert_array_equal(out.outputs_q88, ref.outputs_q88)

    def test_available_backends_marks_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failing backend is reported unavailable."""

        def _boom(*_args: object, **_kwargs: object) -> DclsBatchResult:
            raise OSError("library missing")

        patched = dict(kernel._BACKEND_DISPATCH)
        patched["go"] = _boom
        monkeypatch.setattr(kernel, "_BACKEND_DISPATCH", patched)
        status = available_backends()
        assert status["go"] is False
        assert status["python"] is True
