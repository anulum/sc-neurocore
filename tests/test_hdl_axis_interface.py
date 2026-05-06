# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — AXI-Stream interface HDL tests

from __future__ import annotations

from pathlib import Path


def test_axis_interface_propagates_tlast() -> None:
    hdl = (Path(__file__).resolve().parents[1] / "hdl" / "sc_axis_interface.v").read_text()
    assert "input_last <= s_axis_tlast;" in hdl
    assert "output_last <= input_last;" in hdl
    assert "frame counter for tlast" not in hdl
