# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the unified quasi-random RTL emitter

"""Test suite for QuasiRandomEmitter (Sobol + Halton backends)."""

from __future__ import annotations

import subprocess
import shutil

import pytest

from sc_neurocore.hdl_gen.quasirandom_emitter import (
    Halton16Emitter,
    QuasiRandomEmitter,
)
from sc_neurocore.hdl_gen.sobol16_emitter import Sobol16Emitter


class TestSobol16Emitter:
    """Test Sobol-16 RTL generation."""

    def test_generates_valid_verilog(self) -> None:
        emitter = Sobol16Emitter()
        code = emitter.generate()
        assert "module sc_sobol16_source" in code
        assert "endmodule" in code
        assert "casez" in code

    def test_custom_module_name(self) -> None:
        emitter = Sobol16Emitter(module_name="my_sobol")
        code = emitter.generate()
        assert "module my_sobol" in code

    def test_seed_affects_first_sample(self) -> None:
        e1 = Sobol16Emitter(seed=0)
        e2 = Sobol16Emitter(seed=42)
        assert e1.generate() != e2.generate()


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


class TestQuasiRandomEmitter:
    """Test the unified factory interface."""

    def test_sobol_method(self) -> None:
        emitter = QuasiRandomEmitter(method="sobol")
        code = emitter.generate()
        assert "module sc_sobol16_source" in code
        assert "casez" in code

    def test_halton_method(self) -> None:
        emitter = QuasiRandomEmitter(method="halton")
        code = emitter.generate()
        assert "module sc_halton16_source" in code
        assert "reversed" in code

    def test_default_is_sobol(self) -> None:
        emitter = QuasiRandomEmitter()
        assert emitter.method == "sobol"

    def test_module_name_property(self) -> None:
        emitter = QuasiRandomEmitter(method="halton", module_name="test_qr")
        assert emitter.module_name == "test_qr"

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown quasi-random method"):
            QuasiRandomEmitter(method="mersenne")

    def test_sobol_and_halton_differ(self) -> None:
        sobol_code = QuasiRandomEmitter(method="sobol").generate()
        halton_code = QuasiRandomEmitter(method="halton").generate()
        assert sobol_code != halton_code

    @pytest.mark.skipif(
        not shutil.which("iverilog"),
        reason="Icarus Verilog not installed",
    )
    def test_both_methods_compile(self, tmp_path) -> None:
        for method in ("sobol", "halton"):
            emitter = QuasiRandomEmitter(method=method)
            code = emitter.generate()
            vfile = tmp_path / f"{method}.v"
            vfile.write_text(code)
            result = subprocess.run(
                ["iverilog", "-g2012", "-o", str(tmp_path / f"{method}.out"), str(vfile)],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"iverilog failed for {method}: {result.stderr}"
