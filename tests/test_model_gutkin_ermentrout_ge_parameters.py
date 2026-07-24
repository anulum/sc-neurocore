# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGEParameters from former test_model_gutkin_ermentrout.py

"""Focused suite: TestGEParameters from former test_model_gutkin_ermentrout.py."""

from __future__ import annotations

from tests.model_gutkin_ermentrout_support import *  # noqa: F403


class TestGEParameters:
    @pytest.mark.parametrize("g_na", [10.0, 20.0, 40.0])
    def test_g_na_sweep(self, g_na: float) -> None:
        n = GutkinErmentroutNeuron(g_na=g_na)
        for _ in range(5000):
            n.step(5.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("g_k", [5.0, 10.0, 20.0])
    def test_g_k_sweep(self, g_k: float) -> None:
        n = GutkinErmentroutNeuron(g_k=g_k)
        for _ in range(5000):
            n.step(5.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("dt", [0.02, 0.05, 0.1])
    def test_dt_stability(self, dt: float) -> None:
        n = GutkinErmentroutNeuron(dt=dt)
        for _ in range(10_000):
            n.step(5.0)
        assert np.isfinite(n.v) and np.isfinite(n.n)

    def test_invalid_runtime_current_preserves_state(self) -> None:
        n = GutkinErmentroutNeuron()
        before = (n.v, n.n)
        with pytest.raises(ValueError, match="invalid"):
            n.step(float("nan"))
        assert (n.v, n.n) == before

    def test_invalid_candidate_preserves_state(self) -> None:
        n = GutkinErmentroutNeuron(dt=100.0)
        before = (n.v, n.n)
        with pytest.raises(ValueError, match="candidate"):
            n.step(1.0e9)
        assert (n.v, n.n) == before
