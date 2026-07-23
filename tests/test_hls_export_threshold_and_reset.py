# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestThresholdAndReset from former test_hls_export.py

"""Focused suite: TestThresholdAndReset from former test_hls_export.py."""

from __future__ import annotations

from tests.hls_export_support import *  # noqa: F403

class TestThresholdAndReset:
    def test_threshold_is_configurable(self) -> None:
        cpp = generate_hls_cpp("sc_lif", {"v": "I"}, threshold=2.5)
        assert "const fp_t V_THRESH = fp_t(2.5);" in cpp

    def test_membrane_resets_by_subtracting_threshold(self) -> None:
        cpp = generate_hls_cpp("sc_lif", {"v": "I"})
        assert "v = spike_out ? (v_next - V_THRESH) : v_next;" in cpp

    def test_spike_detection_on_first_state_var(self) -> None:
        cpp = generate_hls_cpp("sc_izh", {"v": "u + I", "u": "v"})
        assert "spike_out = (v_next > V_THRESH);" in cpp
        # The non-membrane variable updates without a reset.
        assert "u = u_next;" in cpp
