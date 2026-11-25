# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ring16 receipt toolchain resolution and child environment

"""The ring16 receipt finds its EDA tools on PATH and runs them in a confined environment.

The receipt used to require every tool under the repository ``.venv/bin``. On ML350 the
shared tools come from ``/opt/eda`` once ``/opt/eda/env.sh`` is sourced, so the receipt
could not be produced there. A tool is now taken from ``.venv/bin`` when present and
otherwise from ``PATH``. Every child process gets the tools' directories first on ``PATH``,
``COLUMNS=4096`` and ``TERM=xterm`` against Yosys's ABC hang on a dumb terminal, and a
``TMPDIR`` inside the run's own temporary directory, where the formal proof also works.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bench() -> ModuleType:
    """Load the benchmark script as a module without running its command line."""
    name = "bench_sc_compte_wm_ring16"
    specification = importlib.util.spec_from_file_location(
        name, ROOT / "benchmarks/bench_sc_compte_wm_ring16.py"
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _executable(directory: Path, name: str) -> Path:
    """Create an executable stand-in named ``name`` in ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_a_tool_in_the_repository_venv_is_preferred(
    bench: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tool present in ``.venv/bin`` is used even when another copy is on PATH."""
    venv = tmp_path / "venv-bin"
    local = _executable(venv, "yosys")
    _executable(tmp_path / "eda", "yosys")
    monkeypatch.setattr(bench, "VENV_BIN", venv)
    monkeypatch.setenv("PATH", str(tmp_path / "eda"))
    assert bench._resolve_tool("yosys") == local
    assert bench._custody(local) == bench.VENV_CUSTODY


def test_a_tool_only_on_path_is_found_there(
    bench: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a ``.venv/bin`` copy the tool is taken from PATH, as after sourcing env.sh."""
    eda = tmp_path / "opt-eda-bin"
    shared = _executable(eda, "cvc5")
    monkeypatch.setattr(bench, "VENV_BIN", tmp_path / "missing-venv-bin")
    monkeypatch.setenv("PATH", str(eda))
    assert bench._resolve_tool("cvc5") == shared
    assert bench._custody(shared) == eda.as_posix()


def test_a_tool_absent_everywhere_is_refused_with_the_remedy(
    bench: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tool neither in ``.venv/bin`` nor on PATH stops the receipt and names the remedy."""
    monkeypatch.setattr(bench, "VENV_BIN", tmp_path / "missing-venv-bin")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    with pytest.raises(RuntimeError, match=r"sby is neither .* source /opt/eda/env.sh"):
        bench._resolve_tool("sby")


def test_the_child_environment_puts_the_tools_first_and_confines_temporary_files(
    bench: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tools' directories lead PATH once each, with a wide terminal and the run's own TMPDIR."""
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("TERM", "dumb")
    tools = {
        "yosys": tmp_path / "a" / "yosys",
        "sby": tmp_path / "a" / "sby",
        "cvc5": tmp_path / "b" / "cvc5",
    }
    environment = bench._tool_environment(tools, tmp_path / "scratch")
    assert environment["PATH"].split(os.pathsep) == [
        str(tmp_path / "a"),
        str(tmp_path / "b"),
        "/usr/bin",
    ]
    assert environment["COLUMNS"] == "4096"
    assert environment["TERM"] == "xterm"
    assert environment["TMPDIR"] == str(tmp_path / "scratch")


def _stand_in_sby(directory: Path, status: str) -> Path:
    """Create a ``sby`` stand-in that records its call and leaves ``status`` in its work directory.

    It writes its arguments, working directory, ``TMPDIR`` and ``COLUMNS`` to ``call.txt`` beside
    itself, then creates the ``-d`` work directory with a ``status`` file, as SymbiYosys does.
    """
    record = directory / "call.txt"
    return _executable_script(
        directory,
        "sby",
        "#!/bin/sh\n"
        f'printf "%s\\n" "$*" "$PWD" "$TMPDIR" "$COLUMNS" > "{record}"\n'
        'while [ "$1" != "-d" ]; do shift; done\n'
        'mkdir -p "$2"\n'
        f'printf "{status}\\n" > "$2/status"\n',
    )


def _executable_script(directory: Path, name: str, body: str) -> Path:
    """Write ``body`` as an executable script named ``name`` in ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_the_formal_proof_works_inside_the_run_directory(bench: ModuleType, tmp_path: Path) -> None:
    """A real ``sby`` process gets the environment, the spec directory and a work dir in scratch."""
    sby = _stand_in_sby(tmp_path / "tools", "PASS 3")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    environment = bench._tool_environment({"sby": sby}, scratch)
    bench._run_formal(sby, environment, scratch)
    arguments, cwd, tmpdir, columns = (
        (tmp_path / "tools" / "call.txt").read_text(encoding="utf-8").splitlines()
    )
    assert arguments == f"-f -d {scratch / 'proof'} {bench.SBY.name}"
    assert Path(cwd) == bench.SBY.parent
    assert tmpdir == str(scratch)
    assert columns == "4096"


def test_a_formal_run_without_pass_is_refused(bench: ModuleType, tmp_path: Path) -> None:
    """A real ``sby`` process that leaves a FAIL status stops the receipt."""
    sby = _stand_in_sby(tmp_path / "tools", "FAIL 2")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with pytest.raises(RuntimeError, match="did not report PASS"):
        bench._run_formal(sby, bench._tool_environment({"sby": sby}, scratch), scratch)
