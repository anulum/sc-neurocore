"""Tests for SpiceGenerator crossbar netlist emission."""

import os
import time

import numpy as np
import pytest

from sc_neurocore.hdl_gen.spice_generator import SpiceGenerator


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def test_spice_generator_writes_file(tmp_path):
    """generate_crossbar should write a netlist file."""
    weights = np.array([[0.0, 1.0]], dtype=float)
    path = tmp_path / "crossbar.sp"
    SpiceGenerator.generate_crossbar(weights, str(path))
    assert path.exists()


def test_spice_generator_header_includes_size(tmp_path):
    """Header should include crossbar dimensions."""
    weights = np.zeros((2, 3), dtype=float)
    path = tmp_path / "crossbar.sp"
    SpiceGenerator.generate_crossbar(weights, str(path))
    text = path.read_text()
    assert "* Memristor Crossbar 2x3" in text


def test_spice_generator_resistance_extremes(tmp_path):
    """Weight extremes should map to expected resistance ranges."""
    weights = np.array([[0.0, 1.0]], dtype=float)
    path = tmp_path / "crossbar.sp"
    SpiceGenerator.generate_crossbar(weights, str(path))
    text = path.read_text()
    assert "1000000.00" in text  # 1 / 1e-6
    assert "10000.00" in text    # 1 / 100e-6


def test_spice_generator_memristor_count(tmp_path):
    """Memristor lines should match rows*cols."""
    weights = np.zeros((3, 2), dtype=float)
    path = tmp_path / "crossbar.sp"
    SpiceGenerator.generate_crossbar(weights, str(path))
    text = path.read_text()
    memristor_lines = [line for line in text.splitlines() if line.startswith("R_")]
    assert len(memristor_lines) == 3 * 2


def test_spice_generator_input_lines(tmp_path):
    """Input lines count should match row count."""
    weights = np.zeros((2, 2), dtype=float)
    path = tmp_path / "crossbar.sp"
    SpiceGenerator.generate_crossbar(weights, str(path))
    text = path.read_text()
    input_lines = [line for line in text.splitlines() if line.startswith("Vin_")]
    assert len(input_lines) == 2


def test_spice_generator_load_lines(tmp_path):
    """Load resistor lines should match column count."""
    weights = np.zeros((2, 4), dtype=float)
    path = tmp_path / "crossbar.sp"
    SpiceGenerator.generate_crossbar(weights, str(path))
    text = path.read_text()
    load_lines = [line for line in text.splitlines() if line.startswith("Rload_")]
    assert len(load_lines) == 4


def test_spice_generator_single_cell(tmp_path):
    """Single weight should yield a 1x1 netlist."""
    weights = np.array([[0.5]], dtype=float)
    path = tmp_path / "crossbar.sp"
    SpiceGenerator.generate_crossbar(weights, str(path))
    text = path.read_text()
    assert "* Memristor Crossbar 1x1" in text


def test_spice_generator_ends_with_end(tmp_path):
    """Netlist should end with .END."""
    weights = np.zeros((1, 1), dtype=float)
    path = tmp_path / "crossbar.sp"
    SpiceGenerator.generate_crossbar(weights, str(path))
    text = path.read_text().strip()
    assert text.endswith(".END")


def test_spice_generator_weight_mapping_mid(tmp_path):
    """Mid weights should map to resistances between extremes."""
    weights = np.array([[0.5]], dtype=float)
    path = tmp_path / "crossbar.sp"
    SpiceGenerator.generate_crossbar(weights, str(path))
    text = path.read_text()
    line = [line for line in text.splitlines() if line.startswith("R_0_0")][0]
    r_val = float(line.split()[-1])
    assert 10000.0 < r_val < 1000000.0


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_spice_generator_perf_small(tmp_path):
    """Benchmark generating a small netlist."""
    weights = np.random.random((10, 10))
    path = tmp_path / "crossbar.sp"
    start = time.perf_counter()
    SpiceGenerator.generate_crossbar(weights, str(path))
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0
