# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHalton16Emitter from former test_quasirandom_emitter.py

"""Focused suite: TestHalton16Emitter from former test_quasirandom_emitter.py."""

from __future__ import annotations

from tests.quasirandom_emitter_support import *  # noqa: F403


class TestHalton16Emitter:
    """Test Halton-16 RTL generation."""

    def test_generates_valid_verilog(self) -> None:
        emitter = Halton16Emitter()
        code = emitter.generate()
        assert "module sc_halton16_source" in code
        assert "endmodule" in code
        assert "reversed" in code

    def test_has_bit_reversal_wiring(self) -> None:
        code = Halton16Emitter().generate()
        # Should have 16 assign statements for bit-reversal
        assert code.count("assign reversed[") == 16

    def test_custom_module_name(self) -> None:
        emitter = Halton16Emitter(module_name="custom_halton")
        code = emitter.generate()
        assert "module custom_halton" in code

    @pytest.mark.skipif(
        not shutil.which("iverilog"),
        reason="Icarus Verilog not installed",
    )
    def test_compiles_with_iverilog(self, tmp_path) -> None:
        code = Halton16Emitter().generate()
        vfile = tmp_path / "halton.v"
        vfile.write_text(code)
        result = subprocess.run(
            ["iverilog", "-g2012", "-o", str(tmp_path / "halton.out"), str(vfile)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"iverilog failed: {result.stderr}"
