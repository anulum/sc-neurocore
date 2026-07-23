# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestProvenanceTrace from former test_explainability.py

"""Focused suite: TestProvenanceTrace from former test_explainability.py."""

from __future__ import annotations

from explainability_support import *  # noqa: F403

class TestProvenanceTrace:
    def test_add_step(self):
        trace = ProvenanceTrace()
        trace.add_step("input", "test data")
        assert trace.num_steps == 1

    def test_finalize(self):
        trace = ProvenanceTrace()
        trace.add_step("input", "data")
        assert not trace.is_complete
        trace.finalize()
        assert trace.is_complete

    def test_chain_hash_deterministic(self):
        t1 = ProvenanceTrace()
        t1.add_step("input", "data")
        t1.add_step("encode", "encoded")
        h1 = t1.chain_hash

        t2 = ProvenanceTrace()
        t2.add_step("input", "data")
        t2.add_step("encode", "encoded")
        h2 = t2.chain_hash
        assert h1 == h2

    def test_chain_hash_changes_on_tamper(self):
        t1 = ProvenanceTrace()
        t1.add_step("input", "data")
        h1 = t1.chain_hash

        t2 = ProvenanceTrace()
        t2.add_step("input", "tampered")
        h2 = t2.chain_hash
        assert h1 != h2

    def test_to_list(self):
        trace = ProvenanceTrace()
        trace.add_step("input", "data", metadata={"key": "value"})
        lst = trace.to_list()
        assert len(lst) == 1
        assert lst[0]["stage"] == "input"
        assert lst[0]["metadata"]["key"] == "value"

    def test_data_hash_from_array(self):
        trace = ProvenanceTrace()
        data = np.array([1, 0, 1], dtype=np.uint8)
        step = trace.add_step("encode", "bits", data=data)
        assert len(step.data_hash) == 16
