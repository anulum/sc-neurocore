# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRustParity from former test_spike_stats_dimensionality.py

"""Focused suite: TestRustParity from former test_spike_stats_dimensionality.py."""

from __future__ import annotations

from tests.spike_stats_dimensionality_support import *  # noqa: F403


@pytest.mark.skipif(not _RUST_AVAILABLE, reason="Rust engine not built")
class TestRustParity:
    def test_parity(self) -> None:
        _parity("rust")
