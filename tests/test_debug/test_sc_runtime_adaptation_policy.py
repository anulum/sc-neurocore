# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAdaptationPolicy from former test_sc_runtime.py

"""Focused suite: TestAdaptationPolicy from former test_sc_runtime.py."""

from __future__ import annotations

import pytest
from sc_neurocore.control.sc_runtime import (
    AdaptationPolicy,
    DecorrelatorType,
    RuntimeConfig,
)


class TestAdaptationPolicy:
    """Choosing the next runtime configuration."""

    def test_high_scc_doubles_length(self) -> None:
        """A high SCC doubles the bitstream length to 512."""
        policy = AdaptationPolicy(scc_high=0.15)
        config = RuntimeConfig(bitstream_length=256)
        new, trigger = policy.decide(config, {"ema_scc": 0.20})
        assert trigger == "high_scc"
        assert new.bitstream_length == 512

    def test_low_scc_halves_length(self) -> None:
        """A low SCC halves the bitstream length to 256."""
        policy = AdaptationPolicy(scc_low=0.05)
        config = RuntimeConfig(bitstream_length=512)
        new, trigger = policy.decide(config, {"ema_scc": 0.03})
        assert trigger == "low_scc"
        assert new.bitstream_length == 256

    def test_low_scc_floor_at_min(self) -> None:
        """A low SCC at the minimum length changes nothing."""
        policy = AdaptationPolicy(scc_low=0.05, min_length=256)
        config = RuntimeConfig(bitstream_length=256)
        new, trigger = policy.decide(config, {"ema_scc": 0.03})
        assert trigger is None

    def test_high_scc_enables_ecc(self) -> None:
        """A high SCC turns error correction on."""
        policy = AdaptationPolicy(scc_high=0.10, ecc_trigger_length=2048)
        config = RuntimeConfig(bitstream_length=2048)
        new, trigger = policy.decide(config, {"ema_scc": 0.20})
        assert new.ecc_enabled is True

    def test_drift_switches_decorrelator(self) -> None:
        """Drift switches the decorrelator to Sobol."""
        policy = AdaptationPolicy(enable_decorrelator_cascade=False)
        config = RuntimeConfig(decorrelator=DecorrelatorType.LFSR)
        new, trigger = policy.decide(config, {"ema_scc": 0.08, "drift_detected": True})
        assert trigger == "decorrelator_drift"
        assert new.decorrelator == DecorrelatorType.SOBOL

    def test_stable_no_adaptation(self) -> None:
        """Stable metrics change nothing."""
        policy = AdaptationPolicy()
        config = RuntimeConfig(bitstream_length=512)
        new, trigger = policy.decide(config, {"ema_scc": 0.10, "drift_detected": False})
        assert trigger is None

    def test_decorrelator_cascade_lfsr_to_sobol(self) -> None:
        """The cascade steps from LFSR to Sobol."""
        policy = AdaptationPolicy(enable_decorrelator_cascade=True)
        config = RuntimeConfig(decorrelator=DecorrelatorType.LFSR)
        new, trigger = policy.decide(config, {"ema_scc": 0.08, "drift_detected": True})
        assert trigger == "decorrelator_cascade"
        assert new.decorrelator == DecorrelatorType.SOBOL

    def test_decorrelator_cascade_sobol_to_halton(self) -> None:
        """The cascade steps from Sobol to Halton."""
        policy = AdaptationPolicy(enable_decorrelator_cascade=True)
        config = RuntimeConfig(decorrelator=DecorrelatorType.SOBOL)
        new, trigger = policy.decide(config, {"ema_scc": 0.08, "drift_detected": True})
        assert trigger == "decorrelator_cascade"
        assert new.decorrelator == DecorrelatorType.HALTON

    def test_decorrelator_cascade_halton_to_hybrid(self) -> None:
        """The cascade steps from Halton to hybrid."""
        policy = AdaptationPolicy(enable_decorrelator_cascade=True)
        config = RuntimeConfig(decorrelator=DecorrelatorType.HALTON)
        new, trigger = policy.decide(config, {"ema_scc": 0.08, "drift_detected": True})
        assert trigger == "decorrelator_cascade"
        assert new.decorrelator == DecorrelatorType.HYBRID

    def test_decorrelator_cascade_hybrid_stays(self) -> None:
        """The hybrid decorrelator is the top of the cascade."""
        policy = AdaptationPolicy(enable_decorrelator_cascade=True)
        config = RuntimeConfig(decorrelator=DecorrelatorType.HYBRID)
        new, trigger = policy.decide(config, {"ema_scc": 0.08, "drift_detected": True})
        assert trigger is None  # already at top of cascade

    def test_next_decorrelator_off_cascade_returns_current(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A decorrelator missing from the cascade is kept."""
        # Guards against the cascade table and the DecorrelatorType enum drifting
        # out of sync: a decorrelator missing from the cascade is left unchanged
        # rather than raising. Simulate the drift by shrinking the cascade.
        import sc_neurocore.control.sc_runtime as sc_runtime_module

        monkeypatch.setattr(sc_runtime_module, "DECORRELATOR_CASCADE", [DecorrelatorType.LFSR])
        result = AdaptationPolicy._next_decorrelator(DecorrelatorType.HYBRID)
        assert result == DecorrelatorType.HYBRID
