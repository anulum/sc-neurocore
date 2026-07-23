# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestParameterOverrides from former test_universal_dsl.py

"""Focused suite: TestParameterOverrides from former test_universal_dsl.py."""

from __future__ import annotations

from tests.universal_dsl_support import *  # noqa: F403

class TestParameterOverrides:
    """Test runtime parameter overrides."""

    def test_override_tau(self) -> None:
        # Faster membrane time constant → more spikes
        slow = UniversalNeuron.from_schema("lif")
        fast = UniversalNeuron.from_schema("lif", parameter_overrides={"tau_m": 2.0})
        slow_spikes = sum(slow.step(I=20.0) for _ in range(200))
        fast_spikes = sum(fast.step(I=20.0) for _ in range(200))
        assert fast_spikes >= slow_spikes

    def test_override_dt(self) -> None:
        neuron = UniversalNeuron.from_schema("lif", dt_override=0.5)
        spikes = sum(neuron.step(I=50.0) for _ in range(400))
        # With dt=0.5 (half the default 1.0), the neuron should still spike
        assert spikes > 0, "LIF with dt_override should still produce spikes"

    def test_zero_escape_rate_dt_override_is_rejected_not_ignored(self) -> None:
        with pytest.raises(ValueError, match="dt"):
            UniversalNeuron.from_schema("escape_rate", dt_override=0.0)
