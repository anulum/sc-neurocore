# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMetaController from former test_meta_plasticity.py

"""Focused suite: TestMetaController from former test_meta_plasticity.py."""

from __future__ import annotations

from meta_plasticity_support import *  # noqa: F403

class TestMetaController:
    def _feed_observations(self, mc, n=20, novelty=0.8, surprise=0.5, gci=0.5):
        for _ in range(n):
            mc.observe({"novelty": novelty, "surprise": surprise, "gci": gci})

    def test_no_op_few_observations(self):
        mc = MetaController()
        mc.observe({"novelty": 0.5})
        signals = mc.decide()
        assert signals[0].signal_type == MetaSignalType.NO_OP

    def test_increase_lr_on_high_novelty(self):
        mc = MetaController()
        self._feed_observations(mc, novelty=0.9, surprise=0.5, gci=0.5)
        signals = mc.decide()
        types = [s.signal_type for s in signals]
        assert MetaSignalType.INCREASE_LR in types

    def test_decrease_lr_on_low_novelty(self):
        mc = MetaController()
        self._feed_observations(mc, novelty=0.1, surprise=0.05, gci=0.5)
        signals = mc.decide()
        types = [s.signal_type for s in signals]
        assert MetaSignalType.DECREASE_LR in types

    def test_apply_increase_lr(self):
        mc = MetaController()
        rules = PlasticityRuleSet()
        old_lr = rules.stdp.lr
        sig = MetaControlSignal(MetaSignalType.INCREASE_LR, magnitude=0.5)
        mc.apply_signals(rules, [sig])
        assert rules.stdp.lr > old_lr

    def test_apply_decrease_lr(self):
        mc = MetaController()
        rules = PlasticityRuleSet()
        old_lr = rules.stdp.lr
        sig = MetaControlSignal(MetaSignalType.DECREASE_LR, magnitude=0.5)
        mc.apply_signals(rules, [sig])
        assert rules.stdp.lr < old_lr

    def test_apply_widen_window(self):
        mc = MetaController()
        rules = PlasticityRuleSet()
        old_tau = rules.stdp.tau_plus
        sig = MetaControlSignal(MetaSignalType.WIDEN_WINDOW, magnitude=5.0)
        mc.apply_signals(rules, [sig])
        assert rules.stdp.tau_plus > old_tau

    def test_apply_narrow_window(self):
        mc = MetaController()
        rules = PlasticityRuleSet()
        rules.stdp.tau_plus = 50.0
        sig = MetaControlSignal(MetaSignalType.NARROW_WINDOW, magnitude=5.0)
        mc.apply_signals(rules, [sig])
        assert rules.stdp.tau_plus < 50.0

    def test_signal_history(self):
        mc = MetaController()
        self._feed_observations(mc, novelty=0.9)
        mc.decide()
        assert len(mc.signal_history) > 0

    def test_widen_window_on_unstable_gci(self):
        mc = MetaController()
        for i in range(10):
            mc.observe({"novelty": 0.5, "surprise": 0.5, "gci": 0.1 if i % 2 else 0.9})
        types = [s.signal_type for s in mc.decide()]
        assert MetaSignalType.WIDEN_WINDOW in types

    def test_no_op_when_metrics_are_mid_range_and_stable(self):
        mc = MetaController()
        self._feed_observations(mc, novelty=0.5, surprise=0.5, gci=0.5)
        assert [s.signal_type for s in mc.decide()] == [MetaSignalType.NO_OP]

    def test_apply_increase_homeostatic(self):
        mc = MetaController()
        rules = PlasticityRuleSet()
        old = rules.homeostatic.gain_adaptation_rate
        mc.apply_signals(rules, [MetaControlSignal(MetaSignalType.INCREASE_HOMEOSTATIC)])
        assert rules.homeostatic.gain_adaptation_rate > old

    def test_apply_reset_stp(self):
        mc = MetaController()
        rules = PlasticityRuleSet()
        rules.stp.u_base = 0.9
        mc.apply_signals(rules, [MetaControlSignal(MetaSignalType.RESET_STP)])
        assert rules.stp.u_base == STPParams().u_base
