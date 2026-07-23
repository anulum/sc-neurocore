# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSymbolicPath from former test_explainability.py

"""Focused suite: TestSymbolicPath from former test_explainability.py."""

from __future__ import annotations

from explainability_support import *  # noqa: F403

class TestSymbolicPath:
    def test_add_and_length(self):
        sp = SymbolicPath()
        sp.add("n0", SpikeDecision.SPIKE, "popcount(60) >= threshold(50)")
        sp.add("n1", SpikeDecision.NO_SPIKE, "popcount(30) < threshold(50)")
        assert sp.length == 2

    def test_to_list(self):
        sp = SymbolicPath()
        sp.add("n0", SpikeDecision.SPIKE, "popcount(60) >= threshold(50)")
        lst = sp.to_list()
        assert lst[0]["neuron"] == "n0"
        assert lst[0]["decision"] == "spike"
        assert "popcount" in lst[0]["reason"]
