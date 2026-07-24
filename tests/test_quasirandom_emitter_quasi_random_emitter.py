# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQuasiRandomEmitter from former test_quasirandom_emitter.py

"""Focused suite: TestQuasiRandomEmitter from former test_quasirandom_emitter.py."""

from __future__ import annotations

from tests.quasirandom_emitter_support import *  # noqa: F403


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
