# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDispatch from former test_mixed_dense_kernel.py

"""Focused suite: TestDispatch from former test_mixed_dense_kernel.py."""

from __future__ import annotations

from tests.mixed_dense_kernel_support import *  # noqa: F403

class TestDispatch:
    """Backend dispatch and availability."""

    @staticmethod
    def _workload() -> tuple[np.ndarray, np.ndarray, int, int]:
        rng = np.random.default_rng(11)
        n_outputs, n_inputs, n_batch = 24, 32, 8
        weights = rng.integers(-2048, 2048, n_outputs * n_inputs, dtype=np.int16)
        inputs = rng.integers(-(1 << 18), 1 << 18, n_batch * n_inputs, dtype=np.int32)
        return weights, inputs, n_outputs, n_inputs

    def test_explicit_python_backend(self) -> None:
        weights, inputs, n_outputs, n_inputs = self._workload()
        ref = mixed_dense_forward_batch_q88_q1616(weights, inputs, n_outputs, n_inputs)
        out = mixed_dense_forward_batch(weights, inputs, n_outputs, n_inputs, backend="python")
        npt.assert_array_equal(out.outputs_q1616, ref.outputs_q1616)

    def test_auto_backend_matches_python(self) -> None:
        weights, inputs, n_outputs, n_inputs = self._workload()
        ref = mixed_dense_forward_batch_q88_q1616(weights, inputs, n_outputs, n_inputs)
        out = mixed_dense_forward_batch(weights, inputs, n_outputs, n_inputs, backend="auto")
        npt.assert_array_equal(out.outputs_q1616, ref.outputs_q1616)
        npt.assert_array_equal(out.overflow, ref.overflow)
        npt.assert_array_equal(out.underflow, ref.underflow)

    def test_unknown_backend_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown backend"):
            mixed_dense_forward_batch([1], [1], 1, 1, backend="cuda")

    def test_available_backends_reports_python(self) -> None:
        status = available_backends()
        assert status["python"] is True
        assert set(status) == set(kernel.FASTEST_FIRST_BACKENDS)

    def test_auto_falls_back_to_python(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(*_args: object, **_kwargs: object) -> MixedDenseBatchResult:
            raise ImportError("forced unavailable")

        patched = dict(kernel._BACKEND_DISPATCH)
        for name in ("rust", "mojo", "julia", "go"):
            patched[name] = _boom
        monkeypatch.setattr(kernel, "_BACKEND_DISPATCH", patched)
        weights, inputs, n_outputs, n_inputs = self._workload()
        ref = mixed_dense_forward_batch_q88_q1616(weights, inputs, n_outputs, n_inputs)
        out = mixed_dense_forward_batch(weights, inputs, n_outputs, n_inputs, backend="auto")
        npt.assert_array_equal(out.outputs_q1616, ref.outputs_q1616)

    def test_available_backends_marks_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(*_args: object, **_kwargs: object) -> MixedDenseBatchResult:
            raise OSError("library missing")

        patched = dict(kernel._BACKEND_DISPATCH)
        patched["mojo"] = _boom
        monkeypatch.setattr(kernel, "_BACKEND_DISPATCH", patched)
        status = available_backends()
        assert status["mojo"] is False
        assert status["python"] is True
