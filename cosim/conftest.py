"""Co-simulation fixtures: Verilator detection, compilation, HDL execution."""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import tempfile

import pytest

HDL_DIR = pathlib.Path(__file__).resolve().parent.parent / "hdl"
BUILD_ROOT = pathlib.Path(__file__).resolve().parent / "build"


@pytest.fixture(scope="session")
def verilator_available() -> bool:
    """Check if Verilator is installed and usable."""
    exe = shutil.which("verilator")
    if exe is None:
        pytest.skip("Verilator not found on PATH - skipping co-sim tests.")
    try:
        result = subprocess.run(
            ["verilator", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            pytest.skip(f"Verilator failed: {result.stderr.strip()}")
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        pytest.skip(f"Verilator not usable: {e}")
    return True


@pytest.fixture(scope="session")
def build_dir() -> pathlib.Path:
    """Session-scoped build directory for compiled artifacts."""
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    return BUILD_ROOT



def compile_and_run_verilator(
    top_module: str,
    hdl_files: list[str],
    testbench: str | None,
    build_dir: pathlib.Path,
    stimuli_file: pathlib.Path | None = None,
    timeout: int = 120,
) -> subprocess.CompletedProcess:
    """Compile HDL with Verilator and run the simulation.

    Args:
        top_module: Name of the top Verilog module.
        hdl_files: List of HDL filenames (relative to hdl/).
        testbench: Optional testbench filename (relative to hdl/).
        build_dir: Directory for build artifacts.
        stimuli_file: Optional stimuli file to copy into build dir.
        timeout: Max seconds for compilation + simulation.

    Returns:
        CompletedProcess with stdout/stderr.
    """
    work_dir = build_dir / top_module
    work_dir.mkdir(parents=True, exist_ok=True)

    # Resolve HDL file paths
    hdl_paths = [str(HDL_DIR / f) for f in hdl_files]
    if testbench:
        hdl_paths.append(str(HDL_DIR / testbench))

    # Copy stimuli if provided
    if stimuli_file and stimuli_file.exists():
        shutil.copy2(stimuli_file, work_dir / stimuli_file.name)

    # Verilate
    verilate_cmd = [
        "verilator",
        "--binary",
        "--timing",
        "-Wall",
        "--top-module",
        top_module,
        "--Mdir",
        str(work_dir / "obj_dir"),
        *hdl_paths,
    ]
    result = subprocess.run(
        verilate_cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(work_dir),
    )
    if result.returncode != 0:
        return result  # compilation failed - caller handles

    # Run simulation
    sim_exe = work_dir / "obj_dir" / f"V{top_module}"
    if not sim_exe.exists():
        # Windows may add .exe
        sim_exe = work_dir / "obj_dir" / f"V{top_module}.exe"
    if not sim_exe.exists():
        result.returncode = -1
        result.stderr += f"\nSimulation binary not found: {sim_exe}"
        return result

    sim_result = subprocess.run(
        [str(sim_exe)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(work_dir),
    )
    return sim_result



def read_results_file(path: pathlib.Path) -> list[dict]:
    """Parse a Verilator results file (space-separated key=value per line)."""
    results = []
    if not path.exists():
        return results
    for line in path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        entry = {}
        for token in line.split():
            if "=" in token:
                k, v = token.split("=", 1)
                entry[k] = int(v) if v.lstrip("-").isdigit() else v
            else:
                entry[token] = True
        results.append(entry)
    return results
