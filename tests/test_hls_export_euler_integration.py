# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEulerIntegration from former test_hls_export.py

"""Focused suite: TestEulerIntegration from former test_hls_export.py."""

from __future__ import annotations

from sc_neurocore.compiler.intelligence.hls_export import (
    generate_hls_cpp,
)


class TestEulerIntegration:
    """Euler integration in HLS exports."""

    def test_derivative_is_euler_integrated(self) -> None:
        """Each derivative is integrated by Euler."""
        cpp = generate_hls_cpp("sc_lif", {"v": "-v / tau + I"})
        assert "fp_t d_v = " in cpp
        assert "fp_t v_next = v + dt * d_v;" in cpp

    def test_input_current_lowered_to_I_t(self) -> None:
        """The input current becomes I_t."""
        cpp = generate_hls_cpp("sc_lif", {"v": "I"})
        assert "fp_t d_v = I_t;" in cpp

    def test_dt_is_a_parameter(self) -> None:
        """The time step is a parameter."""
        cpp = generate_hls_cpp("sc_lif", {"v": "I"})
        assert "fp_t dt," in cpp
