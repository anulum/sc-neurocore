# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRallCableNetwork from former test_model_rall_cable.py

"""Focused suite: TestRallCableNetwork from former test_model_rall_cable.py."""

from __future__ import annotations

from tests.model_rall_cable_support import *  # noqa: F403


class TestRallCableNetwork:
    def test_population_incompatible(self) -> None:
        """RallCableNeuron has array-valued v — Population._sync_voltages
        cannot handle this (expects scalar v). Document this limitation."""
        with pytest.raises((ValueError, TypeError)):
            Population(RallCableNeuron, n=5, label="rall")
