# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBrunelWangIsolation from former test_model_brunel_wang.py

"""Focused suite: TestBrunelWangIsolation from former test_model_brunel_wang.py."""

from __future__ import annotations

from tests.model_brunel_wang_support import *  # noqa: F403

class TestBrunelWangIsolation:
    def test_step_returns_binary(self):
        assert BrunelWangNeuron().step(0.0) in (0, 1)

    def test_state_finite(self):
        n = BrunelWangNeuron()
        for _ in range(10000):
            n.step(1.0)
        assert np.isfinite(n.v)

    def test_reset(self):
        n = BrunelWangNeuron()
        for _ in range(100):
            n.step(1.0)
        n.reset()
        assert np.isfinite(n.v)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("v", np.nan),
            ("tau_m", 0.0),
            ("tau_ref", 0.0),
            ("tau_ampa", 0.0),
            ("tau_nmda_rise", 0.0),
            ("tau_nmda_decay", 0.0),
            ("tau_gaba", 0.0),
            ("g_ampa_ext", -1.0),
            ("g_nmda", -1.0),
            ("C_m", 0.0),
            ("mg_conc", -1.0),
            ("dt", 0.0),
        ],
    )
    def test_rejects_invalid_numerical_configuration(self, field: str, value: float):
        with pytest.raises((ValueError, FloatingPointError)):
            BrunelWangNeuron(**{field: value})

    def test_rejects_invalid_synaptic_input_before_state_mutation(self):
        n = BrunelWangNeuron()
        before = (n.v, n.get_state()["ref_remaining"])
        with pytest.raises(ValueError, match="s_nmda_rec"):
            n.step(1.0, s_nmda_rec=np.inf)
        assert (n.v, n.get_state()["ref_remaining"]) == before

    def test_rejects_corrupted_runtime_state_before_mutation(self):
        n = BrunelWangNeuron()
        n.v = np.inf
        before = (n.v, n.get_state()["ref_remaining"])
        with pytest.raises(FloatingPointError, match="voltage state"):
            n.step(1.0)
        assert (n.v, n.get_state()["ref_remaining"]) == before

    def test_nmda_voltage_factor_saturates_for_extreme_negative_voltage(self):
        n = BrunelWangNeuron()
        assert n._nmda_voltage_dep(-1.0e6) == 0.0
