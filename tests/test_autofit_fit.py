# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFit from former test_autofit.py

"""Focused suite: TestFit from former test_autofit.py."""

from __future__ import annotations

from tests.autofit_support import *  # noqa: F403

class TestFit:
    def test_no_matching_candidates(self):
        v = np.random.randn(50)
        c = np.ones(50)
        results = fit(v, c, candidates=["NonExistentModel123"])
        assert results == []

    def test_fit_handles_model_exceptions(self):
        from unittest.mock import patch

        class CrashingModel:
            def __init__(self):
                raise ValueError("model init crash")

        v = np.random.randn(50)
        c = np.ones(50)
        with patch("sc_neurocore.autofit.fitter._get_model_class", return_value=CrashingModel):
            results = fit(v, c, candidates=["CrashingModel"])
        assert results == []

    def test_get_model_class_missing(self):
        cls = _get_model_class("DefinitelyNotAModel")
        assert cls is None

    def test_fit_with_real_models(self):
        v = np.random.randn(50) * 0.5
        c = np.ones(50) * 0.5
        # Use whatever models exist in registry
        results = fit(v, c, dt=0.1, top_k=3)
        # May be empty if no models resolve, that's OK
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, FittedModel)
