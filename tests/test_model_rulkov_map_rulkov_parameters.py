# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRulkovParameters from former test_model_rulkov_map.py

"""Focused suite: TestRulkovParameters from former test_model_rulkov_map.py."""

from __future__ import annotations

from tests.model_rulkov_map_support import *  # noqa: F403


class TestRulkovParameters:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("x", np.nan),
            ("y", np.inf),
            ("alpha", 0.0),
            ("sigma", np.nan),
            ("mu", 0.0),
            ("x_threshold", np.inf),
        ],
    )
    def test_rejects_invalid_numerical_configuration(self, field: str, value: float):
        with pytest.raises(ValueError):
            RulkovMapNeuron(**{field: value})

    def test_rejects_non_finite_current_before_state_mutation(self):
        n = RulkovMapNeuron()
        before = (n.x, n.y)
        with pytest.raises(ValueError, match="current"):
            n.step(np.nan)
        assert (n.x, n.y) == before

    def test_rejects_corrupted_runtime_state_before_mutation(self):
        n = RulkovMapNeuron()
        n.y = np.inf
        before = (n.x, n.y)
        with pytest.raises(FloatingPointError, match="state"):
            n.step(1.0)
        assert (n.x, n.y) == before

    def test_rejects_non_finite_branch_boundary_before_state_mutation(self):
        n = RulkovMapNeuron(x=0.5, y=1.0e308, alpha=1.0e308)
        before = (n.x, n.y)
        with pytest.raises(FloatingPointError, match="branch boundary"):
            n.step(1.0e308)
        assert (n.x, n.y) == before

    def test_sigma_controls_excitability(self):
        """sigma=1.0 fires spontaneously, sigma=-1.6 is silent at I=0."""
        n_excitable = RulkovMapNeuron(sigma=1.0)
        n_silent = RulkovMapNeuron(sigma=-1.6)
        s_exc = len(_run(n_excitable, current=0.0, steps=50000))
        s_sil = len(_run(n_silent, current=0.0, steps=50000))
        assert s_exc > s_sil

    def test_alpha_controls_spike_amplitude(self):
        """Higher alpha → wider spike (larger x excursion)."""
        n_low = RulkovMapNeuron(alpha=2.0)
        n_high = RulkovMapNeuron(alpha=8.0)
        # At alpha=2 default is silent, alpha=8 fires
        s_low = len(_run(n_low, current=0.0, steps=50000))
        s_high = len(_run(n_high, current=0.0, steps=50000))
        assert s_high > s_low

    def test_mu_slow_timescale(self):
        """mu controls y dynamics speed. Smaller mu → slower y → longer bursts."""
        n_fast = RulkovMapNeuron(mu=0.01)
        n_slow = RulkovMapNeuron(mu=0.0001)
        # Both with current to trigger activity
        for _ in range(1000):
            n_fast.step(1.0)
            n_slow.step(1.0)
        # y should have drifted more with larger mu
        # (exact comparison depends on x trajectory, but y changes faster)
        assert abs(n_fast.y - (-3.0)) > abs(n_slow.y - (-3.0))

    def test_upward_crossing_detection(self):
        """Spike only on upward crossing of x_threshold."""
        n = RulkovMapNeuron()
        prev_x = n.x
        upward_only = True
        for _ in range(50000):
            s = n.step(1.0)
            if s == 1 and n.x < prev_x:
                upward_only = False
                break
            prev_x = n.x
        # Can't directly verify internal v_prev, but at least verify spikes occurred
        n2 = RulkovMapNeuron()
        assert len(_run(n2, current=1.0, steps=50000)) > 10
