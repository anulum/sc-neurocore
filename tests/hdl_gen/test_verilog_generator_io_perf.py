# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Verilog generator I/O and performance tests

"""File persistence, write failure, and performance-gate contracts."""

import time

import pytest

from sc_neurocore.hdl_gen.verilog_generator import VerilogGenerator
from tests.hdl_gen.verilog_generator_support import _perf_enabled
from tests.performance_guard import assert_load_tolerant_throughput


def test_verilog_generator_save_to_file(tmp_path):  # type: ignore[no-untyped-def] # Preserved legacy test AST
    """save_to_file should write the generated Verilog."""
    gen = VerilogGenerator(module_name="save_top")
    gen.add_layer("Dense", "dense0", {"n_neurons": 2})
    path = tmp_path / "top.v"
    gen.save_to_file(str(path))
    assert path.exists()
    contents = path.read_text()
    assert "module save_top" in contents


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_verilog_generator_perf_small():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    """Benchmark generating code for a small network."""
    gen = VerilogGenerator()
    for i in range(5):
        gen.add_layer("Dense", f"dense{i}", {"n_neurons": 8})
    start = time.perf_counter()
    _ = gen.generate()
    elapsed = time.perf_counter() - start
    assert_load_tolerant_throughput(
        label="Verilog generation run",
        observed_per_second=1.0 / elapsed,
        strict_minimum_per_second=1.0,
    )


def test_save_to_file_reraises_oserror(tmp_path):  # type: ignore[no-untyped-def] # Preserved legacy test AST
    """save_to_file should log and re-raise when the target path is unwritable."""
    gen = VerilogGenerator(module_name="io_fail")
    # The parent directory does not exist, so open() raises an OSError.
    bad_path = tmp_path / "missing_subdir" / "out.v"
    with pytest.raises(OSError):
        gen.save_to_file(str(bad_path))
