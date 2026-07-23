# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAssociativeMemory from former test_hdc.py

"""Focused suite: TestAssociativeMemory from former test_hdc.py."""

from __future__ import annotations

from tests.hdc_support import *  # noqa: F403

class TestAssociativeMemory:
    def test_store_and_query(self):
        np.random.seed(42)
        enc = HDCEncoder(dim=5000)
        mem = AssociativeMemory()
        va = enc.generate_random_vector()
        vb = enc.generate_random_vector()
        mem.store("A", va)
        mem.store("B", vb)
        assert mem.query(va) == "A"
        assert mem.query(vb) == "B"

    def test_query_with_noise(self):
        np.random.seed(42)
        enc = HDCEncoder(dim=10000)
        mem = AssociativeMemory()
        v = enc.generate_random_vector()
        mem.store("target", v)
        mem.store("other", enc.generate_random_vector())
        # Add 10% noise
        noisy = v.copy()
        flip = np.random.choice(10000, size=1000, replace=False)
        noisy[flip] = 1 - noisy[flip]
        assert mem.query(noisy) == "target"
