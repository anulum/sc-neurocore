# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Short-term plasticity synapse contracts

"""Module-specific behavioural contracts for ``ShortTermPlasticitySynapse``."""

from __future__ import annotations

import pytest


class TestShortTermPlasticitySynapse:
    @pytest.fixture()
    def depressing(self):
        from sc_neurocore.synapses import ShortTermPlasticitySynapse

        return ShortTermPlasticitySynapse.new_depressing()

    @pytest.fixture()
    def facilitating(self):
        from sc_neurocore.synapses import ShortTermPlasticitySynapse

        return ShortTermPlasticitySynapse.new_facilitating()

    def test_depressing_defaults(self, depressing):
        assert depressing.u_base == 0.5
        assert depressing.tau_d == 200.0

    def test_facilitating_defaults(self, facilitating):
        assert facilitating.u_base == 0.1
        assert facilitating.tau_f == 500.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"x": -0.01},
            {"x": 1.01},
            {"u": -0.01},
            {"u": 1.01},
            {"u_base": 0.0},
            {"u_base": 1.01},
            {"tau_d": 0.0},
            {"tau_f": 0.0},
            {"amplitude": -0.01},
            {"dt": 0.0},
        ],
    )
    def test_rejects_non_physical_stp_parameters(self, kwargs):
        """Tsodyks-Markram resources, utilisation, and constants must be physical."""
        from sc_neurocore.synapses import ShortTermPlasticitySynapse

        with pytest.raises(ValueError):
            ShortTermPlasticitySynapse(**kwargs)

    @pytest.mark.parametrize("pre_spike", [1, 0, "yes", None])
    def test_rejects_non_boolean_stp_spike_flag(self, pre_spike):
        """Presynaptic event input must be an explicit boolean."""
        from sc_neurocore.synapses import ShortTermPlasticitySynapse

        with pytest.raises(TypeError, match="pre_spike"):
            ShortTermPlasticitySynapse().step(pre_spike)

    def test_depression_successive_spikes(self, depressing):
        """Depressing synapse: PSC decreases with rapid pre_spikes."""
        pscs = [depressing.step(True) for _ in range(5)]
        assert pscs[0] > pscs[1] > pscs[2], f"PSCs should decrease: {pscs}"

    def test_facilitation_successive_spikes(self, facilitating):
        """Facilitating synapse: PSC increases with rapid pre_spikes."""
        pscs = [facilitating.step(True) for _ in range(5)]
        # Facilitation makes u grow, but x depletes. For facilitating params,
        # first few PSCs should increase before x depletion dominates.
        assert pscs[1] > pscs[0], f"2nd PSC should exceed 1st: {pscs[:3]}"

    def test_recovery_after_silence(self, depressing):
        """After depletion, silence allows recovery of x toward 1."""
        for _ in range(10):
            depressing.step(True)
        x_depleted = depressing.x
        for _ in range(2000):
            depressing.step(False)
        assert depressing.x > x_depleted + 0.3

    def test_no_spike_no_current(self, depressing):
        """No pre_spike → zero PSC."""
        psc = depressing.step(False)
        assert psc == 0.0

    def test_x_never_negative(self, depressing):
        """Resources x must be clamped >= 0."""
        for _ in range(100):
            depressing.step(True)
        assert depressing.x >= 0.0

    def test_reset(self, depressing):
        for _ in range(10):
            depressing.step(True)
        depressing.reset()
        assert depressing.x == 1.0
        assert depressing.u == depressing.u_base
