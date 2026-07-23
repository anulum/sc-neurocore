# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNeuronState from former test_model_zoo.py

"""Focused suite: TestNeuronState from former test_model_zoo.py."""

from __future__ import annotations

from model_zoo_support import *  # noqa: F403

class TestNeuronState:
    def test_get_set(self):
        s = NeuronState({"V": -65.0})
        assert s["V"] == -65.0
        s["V"] = -50.0
        assert s["V"] == -50.0

    def test_copy_independent(self):
        s = NeuronState({"V": -65.0})
        c = s.copy()
        c["V"] = 0.0
        assert s["V"] == -65.0

    def test_as_dict(self):
        s = NeuronState({"V": -65.0, "u": -14.0})
        d = s.as_dict()
        assert d == {"V": -65.0, "u": -14.0}
