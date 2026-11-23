# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGPUVecMAC from former test_gpu_backend.py

"""Focused suite: TestGPUVecMAC from former test_gpu_backend.py."""

from __future__ import annotations

import numpy as np
import pytest
import sc_neurocore.accel.gpu_backend as gb
from sc_neurocore.accel.gpu_backend import (
    gpu_vec_mac,
    to_host,
    xp,
)
from typing import Never


class TestGPUVecMAC:
    """Bitwise multiply-accumulate of word vectors."""

    def test_simple_mac(self) -> None:
        """A full match counts 64 and none counts 0."""
        # 2 neurons, 1 input, 1 word
        # weights all 1s, input all 1s -> popcount = 64
        w = xp.array([[[0xFFFFFFFFFFFFFFFF]], [[0]]], dtype=xp.uint64)
        inp = xp.array([[0xFFFFFFFFFFFFFFFF]], dtype=xp.uint64)
        result = to_host(gpu_vec_mac(w, inp))
        assert result[0] == 64
        assert result[1] == 0

    def test_multiple_inputs(self) -> None:
        """Two full matches count 128."""
        # 1 neuron, 2 inputs, 1 word each — all 1s
        w = xp.array([[[0xFFFFFFFFFFFFFFFF], [0xFFFFFFFFFFFFFFFF]]], dtype=xp.uint64)
        inp = xp.array([[0xFFFFFFFFFFFFFFFF], [0xFFFFFFFFFFFFFFFF]], dtype=xp.uint64)
        result = to_host(gpu_vec_mac(w, inp))
        assert result[0] == 128  # 64 + 64

    def test_zero_input(self) -> None:
        """A zero input counts nothing."""
        w = xp.ones((4, 3, 2), dtype=xp.uint64) * 0xFFFFFFFFFFFFFFFF
        inp = xp.zeros((3, 2), dtype=xp.uint64)
        result = to_host(gpu_vec_mac(w, inp))
        np.testing.assert_array_equal(result, 0)

    @pytest.mark.skipif(not hasattr(gb, "cp") or not gb.HAS_CUPY, reason="CuPy unavailable")
    def test_runtime_failure_falls_back_to_numpy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A GPU runtime failure falls back to NumPy."""
        cupy_module = vars(gb)["cp"]
        original = cupy_module.bitwise_and
        monkeypatch.setattr(gb, "_GPU_RUNTIME_BROKEN", False)

        def _broken(*args: object, **kwargs: object) -> Never:
            raise RuntimeError("Failed to auto-detect CUDA root directory")

        monkeypatch.setattr(cupy_module, "bitwise_and", _broken)
        w = np.array([[[0xFFFFFFFFFFFFFFFF]], [[0]]], dtype=np.uint64)
        inp = np.array([[0xFFFFFFFFFFFFFFFF]], dtype=np.uint64)
        result = to_host(gpu_vec_mac(w, inp))

        assert result[0] == 64
        assert result[1] == 0
        assert gb._GPU_RUNTIME_BROKEN is True

        monkeypatch.setattr(cupy_module, "bitwise_and", original)
