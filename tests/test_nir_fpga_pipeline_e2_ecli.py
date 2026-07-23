# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestE2ECLI from former test_nir_fpga_pipeline.py

"""Focused suite: TestE2ECLI from former test_nir_fpga_pipeline.py."""

from __future__ import annotations

from tests.nir_fpga_pipeline_support import *  # noqa: F403

class TestE2ECLI:
    """Write .nir file → invoke compile-nir → verify output files."""

    def test_cli_compile_nir(self, local_tmp_path):
        """Full CLI E2E: write NIR file, run compile-nir, check outputs."""
        graph = _build_lif_feedforward(n_in=3, n_hidden=4, n_out=2)

        # Write NIR file
        nir_path = str(local_tmp_path / "test_model.nir")
        nir.write(nir_path, graph)

        out_dir = str(local_tmp_path / "compile_output")

        # Run CLI
        cmd = [
            sys.executable,
            "-m",
            "sc_neurocore.cli",
            "compile-nir",
            nir_path,
            "-o",
            out_dir,
            "--module-name",
            "cli_test_net",
            "--dt",
            "1.0",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        assert proc.returncode == 0, f"CLI failed:\n{proc.stderr}\n{proc.stdout}"

        # Check output files exist
        assert os.path.exists(os.path.join(out_dir, "cli_test_net.v"))
        assert os.path.exists(os.path.join(out_dir, "sc_nir_lif.v"))
        assert os.path.exists(os.path.join(out_dir, "sc_nir_weight_rom.v"))

        # Check files contain valid Verilog
        with open(os.path.join(out_dir, "cli_test_net.v")) as f:
            top = f.read()
        assert "module cli_test_net" in top
        assert "endmodule" in top

        with open(os.path.join(out_dir, "sc_nir_lif.v")) as f:
            lif = f.read()
        assert "module sc_nir_lif" in lif
        assert "spike_out" in lif
