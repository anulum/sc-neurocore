# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBrunelWangIsolation from former test_model_brunel_wang.py

"""Focused suite: TestBrunelWangIsolation from former test_model_brunel_wang.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.neurons.models.brunel_wang import (
    BrunelWangNeuron,
)


class TestBrunelWangIsolation:
    """The BrunelWangNeuron in isolation."""

    def test_step_returns_binary(self) -> None:
        """One step returns a binary spike."""
        assert BrunelWangNeuron().step(0.0) in (0, 1)

    def test_state_finite(self) -> None:
        """The state stays finite under drive."""
        n = BrunelWangNeuron()
        for _ in range(10000):
            n.step(1.0)
        assert np.isfinite(n.v)

    def test_reset(self) -> None:
        """A reset after drive leaves a valid state."""
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
    def test_rejects_invalid_numerical_configuration(self, field: str, value: float) -> None:
        """An invalid numerical configuration is rejected."""
        with pytest.raises((ValueError, FloatingPointError)):
            BrunelWangNeuron(**{field: value})

    def test_rejects_invalid_synaptic_input_before_state_mutation(self) -> None:
        """Invalid synaptic input is rejected without change."""
        n = BrunelWangNeuron()
        before = (n.v, n.get_state()["ref_remaining"])
        with pytest.raises(ValueError, match="s_nmda_rec"):
            n.step(1.0, s_nmda_rec=np.inf)
        assert (n.v, n.get_state()["ref_remaining"]) == before

    def test_rejects_corrupted_runtime_state_before_mutation(self) -> None:
        """Corrupted runtime state is rejected before any change."""
        n = BrunelWangNeuron()
        n.v = np.inf
        before = (n.v, n.get_state()["ref_remaining"])
        with pytest.raises(FloatingPointError, match="voltage state"):
            n.step(1.0)
        assert (n.v, n.get_state()["ref_remaining"]) == before

    def test_nmda_voltage_factor_saturates_for_extreme_negative_voltage(self) -> None:
        """The NMDA voltage factor saturates at extreme negative voltage."""
        n = BrunelWangNeuron()
        assert n._nmda_voltage_dep(-1.0e6) == 0.0
