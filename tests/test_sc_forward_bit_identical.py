# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBitIdentical from former test_sc_forward.py

"""Focused suite: TestBitIdentical from former test_sc_forward.py."""

from __future__ import annotations

from tests.sc_forward_support import *  # noqa: F403


@pytest.mark.skipif(not _RUST_AVAILABLE, reason="rust backend not built in this environment")
class TestBitIdentical:
    """NEU-SCPN.4 — Rust and NumPy backends agree to the last bit for a fixed seed."""

    @pytest.mark.parametrize("shape", [(1, 1), (5, 9), (16, 33)])
    @pytest.mark.parametrize("length", [64, 1024, 4096])
    def test_rust_numpy_exact(self, shape: tuple[int, int], length: int) -> None:
        rng = np.random.default_rng(shape[0] * 1000 + shape[1] + length)
        weights = rng.random(shape)
        probs = rng.random(shape[1])
        packed = _pack_weights(weights, length, seed=0x9999)
        rust = sc_forward(packed, probs, length=length, backend="rust", seed=7)
        numpy_floor = sc_forward(packed, probs, length=length, backend="numpy", seed=7)
        npt.assert_array_equal(rust, numpy_floor)

    def test_rust_backend_rejects_non_3d_weights(self) -> None:
        with pytest.raises(ValueError, match="must be 3-D"):
            sc_forward(np.zeros((1, 1), dtype=np.uint64), np.zeros(1), length=64, backend="rust")
