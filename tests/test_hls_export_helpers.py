# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHelpers from former test_hls_export.py

"""Focused suite: TestHelpers from former test_hls_export.py."""

from __future__ import annotations

from tests.hls_export_support import *  # noqa: F403


class TestHelpers:
    def test_sigmoid_helper_emitted_when_used(self) -> None:
        cpp = generate_hls_cpp("sc_x", {"v": "sigmoid(v)"})
        assert "static inline fp_t sc_sigmoid(fp_t x)" in cpp

    def test_exprel_helper_emitted_when_used(self) -> None:
        cpp = generate_hls_cpp("sc_x", {"v": "exprel(v)"})
        assert "static inline fp_t sc_exprel(fp_t x)" in cpp

    def test_helpers_absent_when_unused(self) -> None:
        cpp = generate_hls_cpp("sc_lif", {"v": "I"})
        assert "sc_sigmoid" not in cpp
        assert "sc_exprel" not in cpp
