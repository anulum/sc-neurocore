# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQuantizationError from former test_quantizer.py

"""Focused suite: TestQuantizationError from former test_quantizer.py."""

from __future__ import annotations

from tests.quantizer_support import *  # noqa: F403

class TestQuantizationError:
    def test_error_stats(self):
        w = np.random.randn(100)
        stats = quantization_error(w, fmt="Q8.8")
        assert stats["max_abs_error"] < 1 / 256 + 1e-9
        assert stats["mean_abs_error"] < 1 / 256
        assert stats["rmse"] > 0
        assert stats["snr_db"] > 30  # good SNR for Q8.8

    def test_higher_precision_lower_error(self):
        w = np.random.randn(100)
        e88 = quantization_error(w, fmt="Q8.8")
        e412 = quantization_error(w, fmt="Q4.12")
        assert e412["rmse"] < e88["rmse"]

    def test_q16_16_dominates_q8_8(self):
        w = np.random.RandomState(0).normal(size=1000)
        e88 = quantization_error(w, fmt="Q8.8")
        e1616 = quantization_error(w, fmt="Q16.16")
        assert e1616["rmse"] < e88["rmse"]
        assert e1616["max_abs_error"] < e88["max_abs_error"]
        assert e1616["snr_db"] > e88["snr_db"]
