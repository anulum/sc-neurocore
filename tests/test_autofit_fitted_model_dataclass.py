# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFittedModelDataclass from former test_autofit.py

"""Focused suite: TestFittedModelDataclass from former test_autofit.py."""

from __future__ import annotations

from tests.autofit_support import *  # noqa: F403

class TestFittedModelDataclass:
    def test_fields(self):
        fm = FittedModel(
            model_name="test",
            model_class=type,
            params={"a": 1},
            rmse=0.5,
            feature_error=0.3,
            combined_score=0.4,
            simulated_voltage=np.zeros(10),
        )
        assert fm.model_name == "test"
        assert fm.rmse == 0.5
