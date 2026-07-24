# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAstrocyteAdapterIsolation from former test_model_astrocyte_adapter.py

"""Focused suite: TestAstrocyteAdapterIsolation from former test_model_astrocyte_adapter.py."""

from __future__ import annotations

from tests.model_astrocyte_adapter_support import *  # noqa: F403


class TestAstrocyteAdapterIsolation:
    """Isolation tests for adapter state and validation contracts."""

    def test_defaults(self) -> None:
        """Default parameters expose resting calcium as pseudo-voltage."""
        n = AstrocyteNeuron()
        assert n.ca_threshold == 0.3
        assert n.dt == 0.01
        assert n.v == n.ca  # v exposes Ca

    def test_step_returns_binary(self) -> None:
        """Adapter converts Ca to int {0,1}."""
        assert AstrocyteNeuron().step(0.0) in (0, 1)

    def test_v_tracks_ca(self) -> None:
        """V attribute mirrors Ca concentration."""
        n = AstrocyteNeuron()
        n.step(0.5)
        assert n.v == n.ca

    def test_ca_property(self) -> None:
        """Ca property delegates to the wrapped astrocyte model."""
        n = AstrocyteNeuron()
        n.step(0.5)
        assert n.ca > 0

    def test_ip3_property(self) -> None:
        """IP3 property delegates to the wrapped astrocyte model."""
        n = AstrocyteNeuron()
        n.step(0.5)
        assert n.ip3 > 0

    def test_state_finite(self) -> None:
        """Long adapter runs keep exposed state finite."""
        n = AstrocyteNeuron()
        for _ in range(50000):
            n.step(0.5)
        assert np.isfinite(n.v) and np.isfinite(n.ca) and np.isfinite(n.ip3)

    def test_reset(self) -> None:
        """Reset restores resting calcium as pseudo-voltage."""
        n = AstrocyteNeuron()
        for _ in range(100):
            n.step(1.0)
        n.reset()
        assert n.v == 0.05  # ca initial

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"ca_threshold": -0.01},
            {"ca_threshold": float("nan")},
            {"ca_threshold": float("inf")},
            {"dt": 0.0},
            {"dt": -0.01},
            {"dt": float("nan")},
            {"dt": float("inf")},
        ],
    )
    def test_rejects_non_physical_adapter_parameters(self, kwargs: dict[str, float]) -> None:
        """Adapter threshold and timestep must be finite physical parameters."""
        with pytest.raises(ValueError):
            AstrocyteNeuron(**kwargs)

    @pytest.mark.parametrize("current", [-0.01, float("nan"), float("inf")])
    def test_rejects_non_physical_adapter_drive(self, current: float) -> None:
        """Adapter must preserve the finite non-negative IP3 drive contract."""
        with pytest.raises(ValueError, match="current"):
            AstrocyteNeuron().step(current)
