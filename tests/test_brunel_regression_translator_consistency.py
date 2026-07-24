# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTranslatorConsistency from former test_brunel_regression.py

"""Focused suite: TestTranslatorConsistency from former test_brunel_regression.py."""

from __future__ import annotations

from tests.brunel_regression_support import *  # noqa: F403


class TestTranslatorConsistency:
    """Variants that share StochasticLIFNeuron base must produce identical neuron_kwargs
    except for the specific parameter they modify."""

    def test_v7_only_changes_noise(self):
        bp = BrunelParams()
        v1 = translate_v1_stochastic_lif(bp)
        v7 = translate_v7_noisy(bp)
        for k in v1["neuron_kwargs"]:
            if k == "noise_std":
                assert v7["neuron_kwargs"][k] == 1.0
            else:
                assert v7["neuron_kwargs"][k] == v1["neuron_kwargs"][k]

    def test_v8_only_changes_refractory(self):
        bp = BrunelParams()
        v1 = translate_v1_stochastic_lif(bp)
        v8 = translate_v8_refractory(bp)
        for k in v1["neuron_kwargs"]:
            v1_val = v1["neuron_kwargs"][k]
            v8_val = v8["neuron_kwargs"].get(k, v1_val)
            assert v8_val == v1_val
        assert v8["neuron_kwargs"]["refractory_period"] == 5

    def test_v9_only_adds_kick_flag(self):
        bp = BrunelParams()
        v1 = translate_v1_stochastic_lif(bp)
        v9 = translate_v9_post_kick(bp)
        assert v9["kick_after_step"] is True
        assert v9["neuron_kwargs"] == v1["neuron_kwargs"]

    def test_v10_exact_leak_factor(self):
        bp = BrunelParams(dt=0.1, tau_mem=20.0)
        v10 = translate_v10_exact_leak(bp)
        assert v10["exact_leak"] is True
        expected = np.exp(-bp.dt / bp.tau_mem)
        assert v10["leak_factor"] == pytest.approx(expected, abs=1e-12)
