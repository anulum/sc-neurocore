# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHeaderErrors from former test_equivalence_miter.py

"""Focused suite: TestHeaderErrors from former test_equivalence_miter.py."""

from __future__ import annotations

from tests.equivalence_miter_support import *  # noqa: F403


class TestHeaderErrors:
    """Defensive parsing paths in the module-header locator."""

    def test_malformed_parameter_block(self) -> None:
        with pytest.raises(ValueError, match="malformed parameter block"):
            parse_module_interface("module m #", "m")

    def test_no_port_list_at_all(self) -> None:
        with pytest.raises(ValueError, match="port list .* not found"):
            parse_module_interface("module m; endmodule", "m")

    def test_unterminated_port_list(self) -> None:
        with pytest.raises(ValueError, match="unterminated port list"):
            parse_module_interface("module m ( input wire a", "m")

    def test_non_positive_width_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-positive width"):
            parse_module_interface("module m(input wire [0:5] a); endmodule", "m")

    def test_parameter_block_before_ports_is_skipped(self) -> None:
        src = "module m #(parameter W = 4)(input wire [W-1:0] a, output wire b); endmodule"
        ports = parse_module_interface(src, "m", params={"W": 4})
        assert {p.name: p.width for p in ports} == {"a": 4, "b": 1}
