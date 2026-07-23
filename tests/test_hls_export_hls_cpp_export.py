# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHLSCppExport from former test_hls_export.py

"""Focused suite: TestHLSCppExport from former test_hls_export.py."""

from __future__ import annotations

from tests.hls_export_support import *  # noqa: F403

class TestHLSCppExport:
    """Vitis/Catapult HLS C++ translation."""

    def test_vitis_export(self) -> None:
        from sc_neurocore.compiler.intelligence import generate_hls_cpp

        cpp = generate_hls_cpp(
            "sc_lif",
            {"v": "v + I_t - v * leak"},
            data_width=16,
            fraction=8,
        )
        assert "ap_fixed<16,8>" in cpp
        assert "#pragma HLS PIPELINE" in cpp
        assert "void sc_lif(" in cpp
        assert "V_THRESH" in cpp

    def test_catapult_export(self) -> None:
        from sc_neurocore.compiler.intelligence import generate_hls_cpp

        cpp = generate_hls_cpp(
            "sc_hh",
            {"v": "a + b", "n": "c * d"},
            hls_tool="catapult",
        )
        assert "Catapult" in cpp
        assert "v_next" in cpp
        assert "n_next" in cpp

    def test_include_guard(self) -> None:
        from sc_neurocore.compiler.intelligence import generate_hls_cpp

        cpp = generate_hls_cpp("sc_lif", {"v": "a + b"})
        assert "#ifndef SC_LIF_HLS_H" in cpp
        assert "#endif" in cpp

    def test_state_struct(self) -> None:
        from sc_neurocore.compiler.intelligence import generate_hls_cpp

        cpp = generate_hls_cpp("sc_izh", {"v": "a", "u": "b"})
        assert "struct sc_izh_state" in cpp
        assert "fp_t v;" in cpp
        assert "fp_t u;" in cpp

    def test_spike_detection(self) -> None:
        from sc_neurocore.compiler.intelligence import generate_hls_cpp

        cpp = generate_hls_cpp("sc_lif", {"v": "v + I_t"})
        assert "spike_out" in cpp
        assert "V_THRESH" in cpp
