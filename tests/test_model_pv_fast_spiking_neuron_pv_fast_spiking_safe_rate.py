# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPVFastSpikingSafeRate from former test_model_pv_fast_spiking_neuron.py

"""Focused suite: TestPVFastSpikingSafeRate from former test_model_pv_fast_spiking_neuron.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.neurons.models.pv_fast_spiking_neuron import (
    _safe_rate,
)


class TestPVFastSpikingSafeRate:
    """The safe gating-rate helper at and around its singularity."""

    def test_fallback_returned_at_singularity(self) -> None:
        """Fallback returned at singularity."""
        # v + vhalf = 0 -> the L'Hôpital limit a*k is returned.
        assert _safe_rate(0.1, 35.0, -35.0, 10.0, 1.0) == 1.0

    def test_regular_branch_matches_hodgkin_huxley_ratio(self) -> None:
        """Away from the singularity the rate matches the Hodgkin-Huxley ratio."""
        v, a, vhalf, k = -30.0, 0.1, 35.0, 10.0
        d = v + vhalf
        expected = a * d / (1.0 - np.exp(-d / k))
        assert _safe_rate(a, vhalf, v, k, 1.0) == pytest.approx(expected)
