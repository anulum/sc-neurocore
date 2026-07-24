# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDiagnoseIntegration from former test_doctor.py

"""Focused suite: TestDiagnoseIntegration from former test_doctor.py."""

from __future__ import annotations

from tests.doctor_support import *  # noqa: F403


class TestDiagnoseIntegration:
    def test_full_diagnosis(self):
        layers = [(64, 32), (32, 10)]
        weights = [np.random.randn(32, 64) * 0.3, np.random.randn(10, 32) * 0.3]
        rates = [np.random.uniform(0.05, 0.3, 32), np.random.uniform(0.05, 0.3, 10)]
        r = diagnose(layers, weights=weights, spike_rates=rates, target="artix7")
        assert isinstance(r, DiagnosticReport)
        assert r.score >= 0
        assert r.score <= 100
        s = r.summary()
        assert "Architecture Doctor" in s
