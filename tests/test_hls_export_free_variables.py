# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFreeVariables from former test_hls_export.py

"""Focused suite: TestFreeVariables from former test_hls_export.py."""

from __future__ import annotations

from tests.hls_export_support import *  # noqa: F403

class TestFreeVariables:
    def test_free_vars_become_inputs(self) -> None:
        cpp = generate_hls_cpp("sc_lif", {"v": "-v / tau + I - leak * v"})
        assert "fp_t tau," in cpp
        assert "fp_t leak," in cpp

    def test_state_vars_not_declared_as_inputs(self) -> None:
        cpp = generate_hls_cpp("sc_izh", {"v": "u + I", "u": "v"})
        # v and u are &-referenced state, never plain input params.
        assert "fp_t &v," in cpp
        assert "fp_t &u," in cpp
