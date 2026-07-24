# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPVFastSpikingSafeRate from former test_model_pv_fast_spiking_neuron.py

"""Focused suite: TestPVFastSpikingSafeRate from former test_model_pv_fast_spiking_neuron.py."""

from __future__ import annotations

from tests.model_pv_fast_spiking_neuron_support import *  # noqa: F403


class TestPVFastSpikingSafeRate:
    def test_fallback_returned_at_singularity(self):
        # v + vhalf = 0 -> the L'Hôpital limit a*k is returned.
        assert _safe_rate(0.1, 35.0, -35.0, 10.0, 1.0) == 1.0

    def test_regular_branch_matches_hodgkin_huxley_ratio(self):
        v, a, vhalf, k = -30.0, 0.1, 35.0, 10.0
        d = v + vhalf
        expected = a * d / (1.0 - np.exp(-d / k))
        assert _safe_rate(a, vhalf, v, k, 1.0) == pytest.approx(expected)
