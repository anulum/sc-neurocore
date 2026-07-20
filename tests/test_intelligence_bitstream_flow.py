# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Open-source bitstream-flow contracts

"""Contracts for generated open-source bitstream build flows."""

from __future__ import annotations

import pytest


class TestOpenSourceMakefile:
    """Open-source FPGA build recipe generation."""

    def test_ice40_recipe_uses_icepack_flow(self) -> None:
        from sc_neurocore.compiler.intelligence import generate_oss_makefile

        makefile = generate_oss_makefile("sc_lif", target="ice40")
        assert "nextpnr-ice40" in makefile
        assert "icepack" in makefile

    def test_ecp5_recipe_uses_ecppack_flow(self) -> None:
        from sc_neurocore.compiler.intelligence import generate_oss_makefile

        makefile = generate_oss_makefile("sc_lif", target="ecp5", device="um5g-85k")
        assert "nextpnr-ecp5 --um5g-85k" in makefile
        assert "ecppack" in makefile

    def test_rejects_unknown_open_source_target(self) -> None:
        from sc_neurocore.compiler.intelligence import generate_oss_makefile

        with pytest.raises(ValueError, match="Unsupported open-source FPGA target"):
            generate_oss_makefile("sc_lif", target="gowin")  # type: ignore[arg-type]
