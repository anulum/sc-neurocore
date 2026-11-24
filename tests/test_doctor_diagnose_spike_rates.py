# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDiagnoseSpikeRates from former test_doctor.py

"""Focused suite: TestDiagnoseSpikeRates from former test_doctor.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.doctor import (
    diagnose,
)
from sc_neurocore.doctor.diagnose import (
    Severity,
)


class TestDiagnoseSpikeRates:
    """Diagnosing neuron firing rates."""

    def test_dead_neurons(self) -> None:
        """Dead neurons are critical."""
        rates = [np.zeros(20)]
        r = diagnose([(10, 20)], spike_rates=rates, target="artix7")
        dead = [f for f in r.findings if f.category == "dead_neurons"]
        assert len(dead) >= 1
        assert dead[0].severity == Severity.CRITICAL

    def test_saturated_neurons(self) -> None:
        """Saturated neurons are reported."""
        rates = [np.ones(10)]
        r = diagnose([(10, 10)], spike_rates=rates, target="artix7")
        sat = [f for f in r.findings if f.category == "saturated_neurons"]
        assert len(sat) >= 1

    def test_healthy_rates(self) -> None:
        """Healthy rates are reported as OK."""
        rates = [np.full(20, 0.15)]
        r = diagnose([(10, 20)], spike_rates=rates, target="artix7")
        ok = [f for f in r.findings if f.category == "spike_efficiency"]
        assert len(ok) >= 1
        assert ok[0].severity == Severity.OK
