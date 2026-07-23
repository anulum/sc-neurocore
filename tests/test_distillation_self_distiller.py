# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSelfDistiller from former test_distillation.py

"""Focused suite: TestSelfDistiller from former test_distillation.py."""

from __future__ import annotations

from tests.distillation_support import *  # noqa: F403

class TestSelfDistiller:
    def test_generate_targets(self):
        sd = SelfDistiller(T_teacher=16, T_student=4)
        targets = sd.generate_targets(lambda x, T: np.random.randn(10), np.zeros(8))
        assert targets.shape == (10,)
        assert abs(targets.sum() - 1.0) < 1e-6
