# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExpIFExponentialEscape from former test_model_expif.py

"""Focused suite: TestExpIFExponentialEscape from former test_model_expif.py."""

from __future__ import annotations

from tests.model_expif_support import *  # noqa: F403


class TestExpIFExponentialEscape:
    def test_exponential_term_at_soft_threshold_equals_delta_t(self) -> None:
        neuron = ExpIFNeuron()
        exponential = neuron.delta_t * math.exp((neuron.v_rh - neuron.v_rh) / neuron.delta_t)
        assert exponential == neuron.delta_t

    def test_hard_cutoff_is_distinct_from_soft_threshold(self) -> None:
        neuron = ExpIFNeuron()
        assert neuron.v_threshold > neuron.v_rh + 20.0 * neuron.delta_t

    def test_rk4_stages_are_bounded_only_at_the_event_surface(self) -> None:
        neuron = ExpIFNeuron()
        assert neuron._rhs(1.0e9, 7.0) == neuron._rhs(neuron.v_threshold, 7.0)
        assert neuron._rhs(neuron.v_threshold - 1.0, 7.0) != neuron._rhs(neuron.v_threshold, 7.0)

    def test_candidate_crossing_cutoff_emits_and_resets(self) -> None:
        neuron = ExpIFNeuron(v=29.0)
        assert neuron.step(0.0) == 1
        assert neuron.v == neuron.v_reset

    def test_delta_t_controls_spike_initiation(self) -> None:
        sharp = len(_run(ExpIFNeuron(delta_t=0.5), current=20.0, steps=10_000))
        broad = len(_run(ExpIFNeuron(delta_t=5.0), current=20.0, steps=10_000))
        assert sharp != broad

    def test_negative_extreme_remains_finite(self) -> None:
        neuron = ExpIFNeuron(v=-1000.0)
        for _ in range(100):
            neuron.step(0.0)
        assert math.isfinite(neuron.v)
