# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_build_accel_backends.py

from __future__ import annotations

"""Support extracted from test_build_accel_backends.py."""

import importlib.util


import subprocess


import sys


from pathlib import Path


from typing import Any


import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_tool() -> Any:
    tool_path = _repo_root() / "tools" / "build_accel_backends.py"
    spec = importlib.util.spec_from_file_location("build_accel_backends", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MOD = _load_tool()


def _make_tree(tmp_path: Path) -> tuple[Path, Path]:
    """Create a miniature accel/package tree covering every discovery branch."""
    accel = tmp_path / "accel"
    models = tmp_path / "models"
    # Go sources: conventional (libtheta.so <- theta.go), renamed via recipe
    # (libhr.so <- hindmarsh_rose.go), and an LGSSM-style world_model backend
    # whose loader uses a pathlib `/` expression rather than os.path.join.
    (accel / "go" / "neurons" / "theta").mkdir(parents=True)
    (accel / "go" / "neurons" / "theta" / "theta.go").write_text("package main\n")
    (accel / "go" / "neurons" / "hindmarsh_rose").mkdir(parents=True)
    (accel / "go" / "neurons" / "hindmarsh_rose" / "hindmarsh_rose.go").write_text("package main\n")
    (accel / "go" / "lgssm").mkdir(parents=True)
    (accel / "go" / "lgssm" / "lgssm.go").write_text("package main\n")
    # A loader output whose source is absent -> must be skipped.
    (accel / "go" / "neurons" / "ghost").mkdir(parents=True)
    # Mojo sources: conventional kernel + a renamed one whose recipe lives only in
    # the .mojo header comment (not any .py), plus the LGSSM world_model source.
    (accel / "mojo" / "kernels").mkdir(parents=True)
    (accel / "mojo" / "kernels" / "theta.mojo").write_text("fn main():\n    pass\n")
    (accel / "mojo" / "neurons").mkdir(parents=True)
    (accel / "mojo" / "neurons" / "hindmarsh_rose.mojo").write_text(
        "# mojo build --emit shared-lib -o libhr.so hindmarsh_rose.mojo\nfn main():\n    pass\n"
    )
    (accel / "mojo" / "world_model").mkdir(parents=True)
    (accel / "mojo" / "world_model" / "lgssm.mojo").write_text("fn main():\n    pass\n")
    # A pruned vendored dir must not be scanned (would crash on non-utf8 / noise).
    (accel / "mojo" / ".pixi").mkdir(parents=True)
    (accel / "mojo" / ".pixi" / "poison.py").write_text("this is not valid python !!!\n")
    models.mkdir()
    (models / "theta.py").write_text(
        "import os\n"
        '_ACCEL_ROOT = "x"\n'
        "def ensure_go_loaded():\n"
        '    p = os.path.join(_ACCEL_ROOT, "go", "neurons", "theta", "libtheta.so")\n'
        "def ensure_mojo_loaded():\n"
        '    q = os.path.join(_ACCEL_ROOT, "mojo", "kernels", "libtheta.so")\n'
        "def unrelated():\n"
        '    return os.path.join("no", "root", "here.so")\n'
    )
    (models / "hindmarsh_rose.py").write_text(
        "import os\n"
        '_ACCEL_ROOT = "x"\n'
        "def ensure_go_loaded():\n"
        '    p = os.path.join(_ACCEL_ROOT, "go", "neurons", "hindmarsh_rose", "libhr.so")\n'
        "def ensure_mojo_loaded():\n"
        '    m = os.path.join(_ACCEL_ROOT, "mojo", "neurons", "libhr.so")\n'
        "# build via: go build -buildmode=c-shared -o libhr.so hindmarsh_rose.go\n"
    )
    # LGSSM-style loader outside neurons/models, using pathlib rooted at PACKAGE_ROOT.
    world = tmp_path / "world_model"
    world.mkdir()
    (world / "_lgssm.py").write_text(
        "from pathlib import Path\n"
        "_PACKAGE_ROOT = Path('x')\n"
        "def _ensure_go_loaded():\n"
        '    p = _PACKAGE_ROOT / "accel" / "go" / "lgssm" / "liblgssm.so"\n'
        "def _ensure_mojo_loaded():\n"
        '    m = _PACKAGE_ROOT / "accel" / "mojo" / "world_model" / "liblgssm.so"\n'
    )
    (models / "ghost.py").write_text(
        "import os\n"
        '_ACCEL_ROOT = "x"\n'
        "def ensure_go_loaded():\n"
        '    p = os.path.join(_ACCEL_ROOT, "go", "neurons", "ghost", "libghost.so")\n'
    )
    return accel, models


def _target(tmp_path: Path, language: str = "go") -> Any:
    src = tmp_path / "src.go"
    src.write_text("package main\n")
    return MOD.BackendTarget(language=language, name="x", source=src, output=tmp_path / "libx.so")


def _fake_completed(returncode: int, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["x"], returncode=returncode, stdout="", stderr=stderr)


def _stub_targets(mod: Any, names: list[str]) -> list[Any]:
    return [
        mod.BackendTarget(
            language="go", name=n, source=Path(f"/{n}.go"), output=Path(f"/lib{n}.so")
        )
        for n in names
    ]



__all__ = ['importlib', 'subprocess', 'sys', 'Path', 'Any', 'pytest', '_repo_root', '_load_tool', 'MOD', '_make_tree', '_target', '_fake_completed', '_stub_targets']
