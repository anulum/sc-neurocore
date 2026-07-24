# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRelayFunction from former test_tvm_lowering.py

"""Focused suite: TestRelayFunction from former test_tvm_lowering.py."""

from __future__ import annotations

from tvm_lowering_support import *  # noqa: F403


class TestRelayFunction(unittest.TestCase):
    def test_to_relay_text(self):
        func = RelayFunction(
            name="test_fn",
            params=[("x", "(128,), dtype=bool")],
            body_lines=["let %y = %x;"],
            ret_var="%y",
            ret_type="(128,), dtype=bool",
        )
        text = func.to_relay_text()
        self.assertIn("def @test_fn", text)
        self.assertIn("%y", text)
