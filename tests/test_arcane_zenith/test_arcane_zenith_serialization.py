# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSerialization from former test_arcane_zenith.py

"""Focused suite: TestSerialization from former test_arcane_zenith.py."""

from __future__ import annotations

from tests.test_arcane_zenith.arcane_zenith_support import *  # noqa: F403

class TestSerialization:
    def test_get_state_contains_neuron_and_four_weights(self):
        core = ArcaneZenithCognitiveCore(backend="torch")
        state = core.get_state()
        # Neuron state keys:
        for key in ("v_fast", "v_work", "v_deep", "confidence", "novelty"):
            assert key in state
        # Plasticity weight keys:
        for key in ("w_tau", "w_nov", "w_conf", "w_lr"):
            assert key in state
            assert isinstance(state[key], float)

    def test_state_dict_roundtrip_restores_four_weights(self):
        src = ArcaneZenithCognitiveCore(backend="torch")
        # Drive the rules so weights diverge from the defaults.
        for _ in range(100):
            src.step(3.0)
        sd = src.get_state_dict()
        assert set(sd.keys()) == {"tau_rule", "nov_rule", "conf_rule", "lr_rule"}

        dst = ArcaneZenithCognitiveCore(backend="torch")
        dst.load_state_dict(sd)
        for name in ("tau_rule", "nov_rule", "conf_rule", "lr_rule"):
            src_w = float(getattr(src, name).get_weights()[0])
            dst_w = float(getattr(dst, name).get_weights()[0])
            assert src_w == pytest.approx(dst_w, abs=1e-6)
