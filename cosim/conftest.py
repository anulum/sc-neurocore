"""
Co-simulation framework configuration.

Generates stimulus files, runs Verilator simulation, and compares
results against the Rust golden model via sc_neurocore_engine.
"""

import pathlib
import subprocess

import pytest

HDL_DIR = pathlib.Path(__file__).parent.parent / "hdl"
COSIM_DIR = pathlib.Path(__file__).parent
BUILD_DIR = COSIM_DIR / "build"


@pytest.fixture(scope="session")
def verilator_available():
    """Check if Verilator is available."""
    try:
        result = subprocess.run(
            ["verilator", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    pytest.skip("Verilator not found - skipping co-sim tests")


@pytest.fixture(scope="session")
def build_dir():
    """Create build directory."""
    BUILD_DIR.mkdir(exist_ok=True)
    return BUILD_DIR


def compile_verilator(top_module: str, sources: list[str], build_dir: pathlib.Path):
    """Compile Verilog sources with Verilator."""
    cmd = [
        "verilator",
        "--cc",
        "--exe",
        "--build",
        "-Wno-fatal",
        f"--Mdir={build_dir / top_module}",
        "-o",
        str(build_dir / top_module / f"V{top_module}"),
    ] + [str(HDL_DIR / s) for s in sources]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    if result.returncode != 0:
        pytest.fail(f"Verilator compilation failed:\n{result.stderr}")
    return build_dir / top_module / f"V{top_module}"
