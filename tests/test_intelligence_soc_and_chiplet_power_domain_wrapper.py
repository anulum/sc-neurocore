# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPowerDomainWrapper from former test_intelligence_soc_and_chiplet.py

"""Focused suite: TestPowerDomainWrapper from former test_intelligence_soc_and_chiplet.py."""

from __future__ import annotations

from tests.intelligence_soc_and_chiplet_support import *  # noqa: F403

class TestPowerDomainWrapper:
    """Clock/power gating wrapper for edge deployment."""

    def test_basic_structure(self):
        from sc_neurocore.compiler.intelligence import (
            generate_power_domain_wrapper,
        )

        v = generate_power_domain_wrapper("sc_lif", data_width=16)
        assert "module sc_lif_pg" in v
        assert "endmodule" in v
        assert "power_down" in v
        assert "power_state" in v

    def test_icg_cell(self):
        from sc_neurocore.compiler.intelligence import (
            generate_power_domain_wrapper,
        )

        v = generate_power_domain_wrapper("sc_lif")
        assert "gated_clk" in v
        assert "clk_enable" in v

    def test_wakeup_counter(self):
        from sc_neurocore.compiler.intelligence import (
            generate_power_domain_wrapper,
        )

        v = generate_power_domain_wrapper("sc_lif", wakeup_cycles=8)
        assert "wakeup_cnt" in v
        assert "active" in v

    def test_state_retention(self):
        from sc_neurocore.compiler.intelligence import (
            generate_power_domain_wrapper,
        )

        v = generate_power_domain_wrapper(
            "sc_izh",
            state_vars=["v", "u"],
        )
        assert "v_out" in v
        assert "u_out" in v
        assert "retain" in v.lower()

    def test_always_on_domain(self):
        from sc_neurocore.compiler.intelligence import (
            generate_power_domain_wrapper,
        )

        v = generate_power_domain_wrapper("sc_lif")
        assert "Always-on" in v
        assert "spike_out" in v
