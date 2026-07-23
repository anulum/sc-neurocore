# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestStructureAndBackends from former test_hls_export.py

"""Focused suite: TestStructureAndBackends from former test_hls_export.py."""

from __future__ import annotations

from tests.hls_export_support import *  # noqa: F403

class TestStructureAndBackends:
    def test_include_guard_and_typedef(self) -> None:
        cpp = generate_hls_cpp("sc_lif", {"v": "I"}, data_width=16, fraction=8)
        assert "#ifndef SC_LIF_HLS_H" in cpp
        assert "#endif // SC_LIF_HLS_H" in cpp
        assert "typedef ap_fixed<16,8> fp_t;" in cpp

    def test_state_struct(self) -> None:
        cpp = generate_hls_cpp("sc_izh", {"v": "u", "u": "v"})
        assert "struct sc_izh_state {" in cpp
        assert "fp_t v;" in cpp
        assert "fp_t u;" in cpp

    def test_vitis_pipeline_pragma(self) -> None:
        cpp = generate_hls_cpp("sc_lif", {"v": "I"}, hls_tool="vitis")
        assert "#pragma HLS PIPELINE II=1" in cpp

    def test_catapult_backend(self) -> None:
        cpp = generate_hls_cpp("sc_lif", {"v": "I"}, hls_tool="catapult")
        assert "Catapult" in cpp
        assert "#pragma HLS PIPELINE" not in cpp
