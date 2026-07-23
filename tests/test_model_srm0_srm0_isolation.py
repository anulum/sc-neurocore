# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSRM0Isolation from former test_model_srm0.py

"""Focused suite: TestSRM0Isolation from former test_model_srm0.py."""

from __future__ import annotations

from tests.model_srm0_support import *  # noqa: F403

class TestSRM0Isolation:
    def test_defaults(self) -> None:
        n = SRM0Neuron()
        assert n.v == 0.0 and n.v_threshold == 1.0 and n.tau_m == 20.0
        assert n.tau_eta == 50.0 and n.eta_reset == 5.0

    def test_step_returns_binary(self) -> None:
        assert SRM0Neuron().step(0.0) in (0, 1)

    def test_state_finite(self) -> None:
        n = SRM0Neuron()
        for _ in range(50000):
            n.step(2.0)
        assert np.isfinite(n.v)

    def test_reset(self) -> None:
        n = SRM0Neuron()
        for _ in range(100):
            n.step(5.0)
        n.reset()
        assert n.v == n.v_rest and n._eta == 0.0

    def test_get_state(self) -> None:
        n = SRM0Neuron()
        n.step(1.0)
        state = n.get_state()
        assert "v" in state and "eta" in state and "t" in state

    @pytest.mark.parametrize(
        "kwargs",
        [{"tau_m": 0.0}, {"tau_eta": -1.0}, {"dt": 0.0}, {"eta_reset": -1.0}, {"v": float("nan")}],
    )
    def test_invalid_initial_contract_rejected(self, kwargs: dict[str, float]) -> None:
        with pytest.raises((TypeError, ValueError)):
            SRM0Neuron(**kwargs)
