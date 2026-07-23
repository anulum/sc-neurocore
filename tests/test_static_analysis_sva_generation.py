# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSVAGeneration from former test_static_analysis.py

"""Focused suite: TestSVAGeneration from former test_static_analysis.py."""

from __future__ import annotations

from tests.static_analysis_support import *  # noqa: F403

class TestSVAGeneration:
    """Test SystemVerilog Assertion generation."""

    def test_basic_sva(self) -> None:
        """Basic SVA should contain module, assertions, and covers."""
        sva = generate_sva(["v"], data_width=16, fraction=8)
        assert "module sc_equation_neuron_sva" in sva
        assert "a_no_overflow_v" in sva
        assert "c_spike_reachable" in sva
        assert "c_no_spike" in sva

    def test_multiple_state_vars(self) -> None:
        """SVA with multiple state variables has assertions for each."""
        sva = generate_sva(["v", "u"], data_width=16, fraction=8)
        assert "a_no_overflow_v" in sva
        assert "a_no_overflow_u" in sva
        assert "c_v_nonzero" in sva
        assert "c_u_nonzero" in sva

    def test_input_bounds(self) -> None:
        """Input assumptions should be generated when bounds are provided."""
        sva = generate_sva(
            ["v"],
            data_width=16,
            fraction=8,
            input_bounds={"I_t": (-1000, 25600)},
        )
        assert "m_I_t_bound" in sva
        assert "assume property" in sva

    def test_stability_check(self) -> None:
        """Stability assertions should be present."""
        sva = generate_sva(["v"], data_width=16, fraction=8)
        assert "a_v_not_stuck_max" in sva
        assert "[*100]" in sva

    def test_custom_module_name(self) -> None:
        """Custom module name should be used."""
        sva = generate_sva(
            ["v"],
            module_name="sc_lif_loihi",
            data_width=24,
            fraction=12,
        )
        assert "sc_lif_loihi_sva" in sva

    def test_unsigned_sva(self) -> None:
        """Unsigned format should not use $signed."""
        sva = generate_sva(["v"], data_width=16, fraction=8, signed=False)
        assert "65535" in sva  # unsigned max

    def test_bind_directive(self) -> None:
        """Should include a commented bind directive."""
        sva = generate_sva(["v"], module_name="sc_lif")
        assert "bind sc_lif" in sva

    def test_do254_header(self) -> None:
        """Should reference DO-254 / IEC 61508."""
        sva = generate_sva(["v"])
        assert "DO-254" in sva
        assert "IEC 61508" in sva
