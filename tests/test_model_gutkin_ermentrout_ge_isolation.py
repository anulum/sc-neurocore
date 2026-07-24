# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGEIsolation from former test_model_gutkin_ermentrout.py

"""Focused suite: TestGEIsolation from former test_model_gutkin_ermentrout.py."""

from __future__ import annotations

from tests.model_gutkin_ermentrout_support import *  # noqa: F403


class TestGEIsolation:
    def test_defaults(self) -> None:
        n = GutkinErmentroutNeuron()
        assert n.v == -65.0 and n.n == 0.1
        assert n.g_na == 20.0 and n.g_k == 10.0 and n.g_l == 8.0
        assert n.dt == 0.05

    def test_step_returns_binary(self) -> None:
        assert GutkinErmentroutNeuron().step(0.0) in (0, 1)

    def test_state_finite_long_run(self) -> None:
        n = GutkinErmentroutNeuron()
        for _ in range(100_000):
            n.step(5.0)
        assert np.isfinite(n.v) and np.isfinite(n.n)

    def test_reset_restores_defaults(self) -> None:
        n = GutkinErmentroutNeuron()
        for _ in range(5000):
            n.step(5.0)
        n.reset()
        assert n.v == -65.0 and n.n == 0.1

    def test_deterministic(self) -> None:
        traces = []
        for _ in range(2):
            n = GutkinErmentroutNeuron()
            trace = [(n.step(5.0), n.v, n.n) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"dt": 0.0}, "invalid"),
            ({"dt": float("nan")}, "invalid"),
            ({"n": -0.1}, "invalid"),
            ({"n": 1.1}, "invalid"),
            ({"g_na": -1.0}, "invalid"),
            ({"v": float("inf")}, "invalid"),
        ],
    )
    def test_invalid_initial_contract_rejected(self, kwargs: dict[str, float], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            GutkinErmentroutNeuron(**kwargs)
