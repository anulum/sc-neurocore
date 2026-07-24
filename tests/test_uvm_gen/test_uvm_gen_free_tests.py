# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Module-level tests from former test_uvm_gen.py

"""Module-level tests from former test_uvm_gen.py."""

from __future__ import annotations

from uvm_gen_support import *  # noqa: F403


def test_from_verilog_source_handles_paramless_module_and_blank_port_entries():
    # No `#(...)` block exercises the parameter-less port-section branch, and the
    # stray comma yields a blank port entry that must be skipped, not parsed.
    module = RTLModule.from_verilog_source(PARAMLESS_VERILOG_WITH_BLANK_PORT)

    assert module.name == "sc_paramless"
    port_names = {port.name for port in module.ports}
    assert {"clk", "done"} <= port_names
    assert "" not in port_names
