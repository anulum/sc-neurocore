# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bioware plasticity tests

"""Tests for biological and homeostatic plasticity adapters."""

from __future__ import annotations

from sc_neurocore.bioware.bioware import BCMPlasticity, BiologicalSTDP, HomeostaticPlasticity


class TestBiologicalSTDP:
    def test_potentiation(self) -> None:
        stdp = BiologicalSTDP()
        dw = stdp.compute_dw(5.0)  # post after pre
        assert dw > 0

    def test_depression(self) -> None:
        stdp = BiologicalSTDP()
        dw = stdp.compute_dw(-5.0)  # pre after post
        assert dw < 0

    def test_zero_dt(self) -> None:
        stdp = BiologicalSTDP()
        assert stdp.compute_dw(0.0) == 0.0

    def test_exponential_decay(self) -> None:
        stdp = BiologicalSTDP(tau_plus_ms=20.0)
        dw_near = stdp.compute_dw(1.0)
        dw_far = stdp.compute_dw(40.0)
        assert abs(dw_near) > abs(dw_far)

    def test_update_weight_bounded(self) -> None:
        stdp = BiologicalSTDP(w_max_q88=512, w_min_q88=0)
        w = stdp.update_weight(500, 1.0)
        assert w <= 512
        w = stdp.update_weight(5, -100.0)
        assert w >= 0


# ── BCMPlasticity Tests ──────────────────────────────────────────────


class TestBCMPlasticity:
    def test_threshold_update(self) -> None:
        bcm = BCMPlasticity()
        bcm.update_theta(10.0, dt_ms=10.0)
        assert bcm.theta > 0

    def test_ltp_above_threshold(self) -> None:
        bcm = BCMPlasticity()
        bcm.theta = 5.0
        dw = bcm.compute_dw(10.0, 10.0)  # post > theta
        assert dw > 0

    def test_ltd_below_threshold(self) -> None:
        bcm = BCMPlasticity()
        bcm.theta = 20.0
        dw = bcm.compute_dw(10.0, 10.0)  # post < theta
        assert dw < 0

    def test_weight_bounded(self) -> None:
        bcm = BCMPlasticity(w_max_q88=512, w_min_q88=0)
        bcm.theta = 0.0
        w = bcm.update_weight(510, 100.0, 100.0)
        assert w <= 512


# ── CultureHealth Tests ─────────────────────────────────────────────


class TestHomeostaticPlasticity:
    def test_at_target_no_change(self) -> None:
        hp = HomeostaticPlasticity(target_rate_hz=10.0)
        new = hp.update_threshold(256, observed_rate_hz=10.0, dt_ms=100.0)
        assert new == 256

    def test_too_fast_increases_threshold(self) -> None:
        hp = HomeostaticPlasticity(target_rate_hz=10.0, tau_homeo_ms=1000.0)
        new = hp.update_threshold(256, observed_rate_hz=50.0, dt_ms=1000.0)
        assert new > 256

    def test_too_slow_decreases_threshold(self) -> None:
        hp = HomeostaticPlasticity(target_rate_hz=10.0, tau_homeo_ms=1000.0)
        new = hp.update_threshold(256, observed_rate_hz=1.0, dt_ms=1000.0)
        assert new < 256

    def test_bounded(self) -> None:
        hp = HomeostaticPlasticity(max_threshold_q88=512, min_threshold_q88=64)
        new = hp.update_threshold(500, observed_rate_hz=1000.0, dt_ms=10000.0)
        assert new <= 512
        new = hp.update_threshold(70, observed_rate_hz=0.0, dt_ms=10000.0)
        assert new >= 64


# ── BioHybridFrameResult — dataclass + mapping dual interface ──────────
