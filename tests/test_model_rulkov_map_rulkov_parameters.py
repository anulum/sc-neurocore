# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRulkovParameters from former test_model_rulkov_map.py

"""Focused suite: TestRulkovParameters from former test_model_rulkov_map.py."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.rulkov_map import RulkovMapNeuron
from tests.model_rulkov_map_support import _run


class TestRulkovParameters:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("x", np.nan),
            ("y", np.inf),
            ("alpha", 0.0),
            ("sigma", np.nan),
            ("mu", 0.0),
        ],
    )
    def test_rejects_invalid_numerical_configuration(self, field: str, value: float) -> None:
        with pytest.raises(ValueError):
            RulkovMapNeuron(**{field: value})

    def test_rejects_non_finite_current_before_state_mutation(self) -> None:
        n = RulkovMapNeuron()
        before = (n.x, n.y)
        with pytest.raises(ValueError, match="current"):
            n.step(np.nan)
        assert (n.x, n.y) == before

    def test_rejects_corrupted_runtime_state_before_mutation(self) -> None:
        n = RulkovMapNeuron()
        n.y = np.inf
        before = (n.x, n.y)
        with pytest.raises(FloatingPointError, match="state"):
            n.step(1.0)
        assert (n.x, n.y) == before

    def test_rejects_non_finite_branch_boundary_before_state_mutation(self) -> None:
        n = RulkovMapNeuron(x=0.5, y=1.0e308, alpha=1.0e308)
        before = (n.x, n.y)
        with pytest.raises(FloatingPointError, match="branch boundary"):
            n.step(1.0e308)
        assert (n.x, n.y) == before

    def test_sigma_controls_excitability(self) -> None:
        """sigma=1.0 fires spontaneously, sigma=-1.6 is silent at I=0."""
        n_excitable = RulkovMapNeuron(sigma=1.0)
        n_silent = RulkovMapNeuron(sigma=-1.6)
        s_exc = len(_run(n_excitable, current=0.0, steps=50000))
        s_sil = len(_run(n_silent, current=0.0, steps=50000))
        assert s_exc > s_sil

    def test_alpha_controls_spike_amplitude(self) -> None:
        """Higher alpha → wider spike (larger x excursion)."""
        n_low = RulkovMapNeuron(alpha=2.0)
        n_high = RulkovMapNeuron(alpha=8.0)
        # At alpha=2 default is silent, alpha=8 fires
        s_low = len(_run(n_low, current=0.0, steps=50000))
        s_high = len(_run(n_high, current=0.0, steps=50000))
        assert s_high > s_low

    def test_mu_slow_timescale(self) -> None:
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

    def test_source_reset_branch_event(self) -> None:
        """An event marks the pre-update rightmost reset branch."""
        n = RulkovMapNeuron()
        event_count = 0
        for _ in range(50000):
            x_previous = n.x
            boundary = n.alpha + n.y + 1.0
            event = n.step(1.0)
            assert event == int(x_previous > 0.0 and x_previous >= boundary)
            if event:
                assert n.x == -1.0
                event_count += 1
        assert event_count > 10
