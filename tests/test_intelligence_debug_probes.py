# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Debug-probe insertion contracts

"""Contracts for compiler debug-probe insertion."""

from __future__ import annotations


class TestDebugProbes:
    def test_xilinx(self) -> None:
        from sc_neurocore.compiler.intelligence import insert_debug_probes

        p = insert_debug_probes("sc_lif", {"v": "a"})
        assert p.probe_type == "ila"
        assert "v" in p.signals
        assert "create_debug_core" in p.tcl_commands

    def test_intel(self) -> None:
        from sc_neurocore.compiler.intelligence import insert_debug_probes

        p = insert_debug_probes("sc_lif", {"v": "a"}, vendor="intel")
        assert p.probe_type == "signaltap"
