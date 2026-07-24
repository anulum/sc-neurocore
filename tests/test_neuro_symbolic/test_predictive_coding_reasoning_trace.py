# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestReasoningTrace from former test_predictive_coding.py

"""Focused suite: TestReasoningTrace from former test_predictive_coding.py."""

from __future__ import annotations

from predictive_coding_support import *  # noqa: F403


class TestReasoningTrace:
    def test_empty_trace(self):
        trace = ReasoningTrace()
        assert trace.length == 0
        assert trace.mean_confidence == 0.0
        assert not trace.is_complete

    def test_add_steps(self):
        trace = ReasoningTrace()
        trace.add("cat", "match", 0.8, 0.9)
        trace.add("dog", "match", 0.3, 0.4)
        assert trace.length == 2
        assert abs(trace.mean_confidence - 0.65) < 0.01

    def test_finalize_marks_complete(self):
        trace = ReasoningTrace(start_ns=1)
        trace.add("x", "op", 0.5, 0.5)
        trace.finalize()
        assert trace.is_complete
        assert trace.end_ns > 0

    def test_to_dict_structure(self):
        trace = ReasoningTrace(start_ns=1)
        trace.add("sym", "op", 0.7, 0.8)
        trace.finalize()
        d = trace.to_dict()
        assert "steps" in d
        assert d["length"] == 1
        assert d["complete"] is True
        assert d["steps"][0]["symbol"] == "sym"
