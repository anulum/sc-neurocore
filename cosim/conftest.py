# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Co-simulation pytest fixtures

"""Co-simulation fixtures: Verilator detection, compilation, HDL execution."""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import tempfile

import pytest

HDL_DIR = pathlib.Path(__file__).resolve().parent.parent / "hdl"
BUILD_ROOT = pathlib.Path(__file__).resolve().parent / "build"
TOOLS_BIN = pathlib.Path(__file__).resolve().parent.parent / ".tools" / "perl" / "c" / "bin"

_VENV_BIN_DIR = "Scripts" if os.name == "nt" else "bin"
_VENV_LIB_DIR = "Lib" if os.name == "nt" else "lib"
_EXE_SUFFIX = ".exe" if os.name == "nt" else ""

VENV_SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / ".venv" / _VENV_BIN_DIR
VENV_VERILATOR = VENV_SCRIPTS / f"verilator{_EXE_SUFFIX}"
VENV_VERILATOR_ROOT = (
    pathlib.Path(__file__).resolve().parent.parent
    / ".venv"
    / _VENV_LIB_DIR
    / "site-packages"
    / "verilator"
)


def _find_sh_dir() -> pathlib.Path | None:
    """Find a directory containing sh (needed by Verilator on Windows)."""
    sh = shutil.which("sh")
    if sh is not None:
        return pathlib.Path(sh).parent
    # Common fallback locations on Windows
    if os.name == "nt":
        for candidate in [
            pathlib.Path(r"C:\Progra~1\Git\usr\bin"),
            pathlib.Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
            / "Git"
            / "usr"
            / "bin",
        ]:
            if (candidate / "sh.exe").exists():
                return candidate
    return None


@pytest.fixture(scope="session")
def verilator_available() -> bool:
    """Check if Verilator is installed and usable."""
    env = _build_subprocess_env()
    exe = _resolve_verilator_executable(env)
    if exe is None:
        pytest.skip("Verilator not found on PATH - skipping co-sim tests.")
    try:
        result = subprocess.run(
            [exe, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        if result.returncode != 0:
            pytest.skip(f"Verilator failed: {result.stderr.strip()}")
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        pytest.skip(f"Verilator not usable: {e}")
    return True


@pytest.fixture(scope="session")
def build_dir() -> pathlib.Path:
    """Session-scoped short build directory for compiled artifacts."""
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    return pathlib.Path(tempfile.mkdtemp(prefix="scn_cosim_"))


def _build_subprocess_env() -> dict[str, str]:
    """Create subprocess env with local toolchain fallbacks (cross-platform)."""
    env = os.environ.copy()
    path_entries: list[str] = []
    sh_dir = _find_sh_dir()
    if os.name == "nt" and sh_dir is not None:
        path_entries.append(str(sh_dir))
    if VENV_SCRIPTS.exists():
        path_entries.append(str(VENV_SCRIPTS))
    if TOOLS_BIN.exists():
        path_entries.append(str(TOOLS_BIN))
    if path_entries:
        env["PATH"] = os.pathsep.join(path_entries + [env.get("PATH", "")])
    if "VERILATOR_ROOT" not in env and VENV_VERILATOR_ROOT.exists():
        env["VERILATOR_ROOT"] = VENV_VERILATOR_ROOT.as_posix()
    if os.name == "nt" and sh_dir is not None:
        sh_exe = sh_dir / "sh.exe"
        if sh_exe.exists():
            env["SHELL"] = str(sh_exe)
            env["MAKESHELL"] = str(sh_exe)
            env.setdefault("MSYS2_ARG_CONV_EXCL", "*")
            env.setdefault("MSYS_NO_PATHCONV", "1")
    return env


def _resolve_verilator_executable(env: dict[str, str]) -> str | None:
    """Resolve a runnable verilator executable from PATH or local venv."""
    exe = shutil.which("verilator", path=env.get("PATH"))
    if exe is not None:
        return exe
    if VENV_VERILATOR.exists():
        return str(VENV_VERILATOR)
    return None


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

    # Resolve HDL file paths using relative POSIX-style paths for Windows toolchain compatibility.
    hdl_paths = [pathlib.Path(os.path.relpath(HDL_DIR / f, work_dir)).as_posix() for f in hdl_files]
    if testbench:
        hdl_paths.append(pathlib.Path(os.path.relpath(HDL_DIR / testbench, work_dir)).as_posix())

    # Copy stimuli if provided
    if stimuli_file and stimuli_file.exists():
        shutil.copy2(stimuli_file, work_dir / stimuli_file.name)

    env = _build_subprocess_env()
    verilator_exe = _resolve_verilator_executable(env) or "verilator"

    # Verilate
    verilate_cmd = [
        verilator_exe,
        "--binary",
        "--timing",
        "-Wno-fatal",
        "-Wno-WIDTHTRUNC",
        "-Wno-WIDTHEXPAND",
        "-CFLAGS",
        "-fcoroutines",
        "--top-module",
        top_module,
        "--Mdir",
        "obj_dir",
        *hdl_paths,
    ]
    result = subprocess.run(
        verilate_cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(work_dir),
        env=env,
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
        env=env,
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
