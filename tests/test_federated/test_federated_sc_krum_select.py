# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestKrumSelect from former test_federated_sc.py

"""Focused suite: TestKrumSelect from former test_federated_sc.py."""

from __future__ import annotations

from federated_sc_support import *  # noqa: F403

class TestKrumSelect:
    def test_selects_central(self):
        vecs = [
            np.array([0.0, 0.0]),
            np.array([0.1, 0.1]),
            np.array([10.0, 10.0]),
        ]
        idx = krum_select(vecs, num_byzantine=1)
        assert idx in [0, 1]

    def test_single_byzantine(self):
        honest = [
            np.array([1.0, 1.0]) + np.random.default_rng(i).standard_normal(2) * 0.1
            for i in range(5)
        ]
        byzantine = [np.array([100.0, -100.0])]
        all_vecs = honest + byzantine
        idx = krum_select(all_vecs, num_byzantine=1)
        assert idx < 5
