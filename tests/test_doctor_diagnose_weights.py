# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDiagnoseWeights from former test_doctor.py

"""Focused suite: TestDiagnoseWeights from former test_doctor.py."""

from __future__ import annotations

from tests.doctor_support import *  # noqa: F403


class TestDiagnoseWeights:
    def test_sparse_weights(self):
        w = [np.zeros((10, 10))]
        r = diagnose([(10, 10)], weights=w, target="artix7")
        sparse = [f for f in r.findings if f.category == "weight_sparsity"]
        assert len(sparse) >= 1

    def test_outlier_weights(self):
        w = [np.ones((10, 10)) * 0.01]
        w[0][0, 0] = 100.0
        r = diagnose([(10, 10)], weights=w, target="artix7")
        outlier = [f for f in r.findings if f.category == "weight_outliers"]
        assert len(outlier) >= 1

    def test_sc_range_warning(self):
        w = [np.random.randn(5, 5) * 5]
        r = diagnose([(5, 5)], weights=w, target="artix7")
        sc = [f for f in r.findings if f.category == "weight_sc_range"]
        assert len(sc) >= 1

    def test_healthy_weights(self):
        w = [np.random.rand(10, 10) * 0.5]
        r = diagnose([(10, 10)], weights=w, target="artix7")
        problems = [
            f for f in r.findings if f.category.startswith("weight") and f.severity != Severity.OK
        ]
        assert len(problems) == 0
