# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Importing a package must not provision an optional runtime

"""Importing a public package must not provision an optional runtime.

Importing ``juliacall`` makes ``juliapkg`` resolve a Julia environment and, on
a machine that has none, install one. Two modules did that at import time — the
cortical column discovered its native accelerators and the gamma-oscillation
circuit loaded its Julia kernel while their modules were being read — and both
are imported by ``sc_neurocore.network``. So a caller who imported the public
network API acquired a Julia toolchain, silently, without asking for one.

Every case here fails on that former behaviour. Each import runs in its own
interpreter with an empty ``juliapkg`` project and offline resolution, so a
module that reaches for Julia cannot quietly succeed against the environment
this checkout already has: it either leaves ``juliacall`` unimported, or the
case fails.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# The public surfaces a caller imports without asking for any native runtime.
PUBLIC_IMPORTS = (
    "sc_neurocore",
    "sc_neurocore.network",
    "sc_neurocore.accel",
    "sc_neurocore.neurons.models",
    "sc_neurocore.studio.models",
    "sc_neurocore.studio.state_layout",
)

# Facades that dispatch to a native lane but must not load one to be imported.
LANE_FACADE_IMPORTS = (
    "sc_neurocore.accel.escape_rate",
    "sc_neurocore.accel.brunel_wang",
    "sc_neurocore.accel.coba_lif",
    "sc_neurocore.network.cortical_column",
    "sc_neurocore.network.gamma_oscillation",
)

PROBE = """
import json
import sys

import {module}  # noqa: F401

print(json.dumps(sorted(name for name in sys.modules if name.split(".")[0] in ("juliacall", "juliapkg"))))
"""


def _import_in_a_bare_interpreter(module: str, project: Path) -> tuple[list[str], str]:
    """Import *module* in its own interpreter and report the Julia modules it loaded.

    The child gets an empty ``juliapkg`` project and offline resolution, so a
    module that reaches for Julia cannot borrow the environment this checkout
    has already provisioned.
    """
    environment = dict(os.environ)
    environment["PYTHON_JULIAPKG_PROJECT"] = str(project)
    environment["PYTHON_JULIAPKG_OFFLINE"] = "yes"
    completed = subprocess.run(
        [sys.executable, "-c", PROBE.format(module=module)],
        capture_output=True,
        text=True,
        env=environment,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    loaded: list[str] = json.loads(lines[-1])
    return loaded, completed.stdout + completed.stderr


@pytest.mark.parametrize("module", PUBLIC_IMPORTS + LANE_FACADE_IMPORTS)
def test_importing_a_package_loads_no_julia_runtime(module: str, tmp_path: Path) -> None:
    """No public import may load ``juliacall``, whose import provisions Julia."""
    loaded, _ = _import_in_a_bare_interpreter(module, tmp_path / "juliapkg")
    assert loaded == []


@pytest.mark.parametrize("module", PUBLIC_IMPORTS)
def test_importing_a_package_writes_no_juliapkg_project(module: str, tmp_path: Path) -> None:
    """Nothing may be written into a fresh environment by an import alone."""
    project = tmp_path / "juliapkg"
    _import_in_a_bare_interpreter(module, project)
    assert not project.exists() or list(project.iterdir()) == []


def test_the_deferred_discovery_still_finds_the_accelerators() -> None:
    """Deferring discovery must not lose it: asking still populates the handles."""
    from sc_neurocore.network import cortical_column

    cortical_column._ensure_native_backends()
    assert cortical_column._BACKENDS_DISCOVERED is True

    handles = (
        cortical_column._HAS_RUST_CSR_SPMV,
        cortical_column._HAS_RUST_CSR_MULTI_SPMV,
        cortical_column._HAS_JULIA_MULTI_SPMV,
        cortical_column._HAS_GO_MULTI_SPMV,
        cortical_column._HAS_MOJO_MULTI_SPMV,
    )
    assert all(isinstance(flag, bool) for flag in handles)


def test_discovery_runs_at_most_once_and_keeps_a_pinned_handle() -> None:
    """A caller that pinned a handle keeps it; discovery never runs twice."""
    from sc_neurocore.network import cortical_column

    cortical_column._ensure_native_backends()
    previous = cortical_column._HAS_RUST_CSR_SPMV
    try:
        cortical_column._HAS_RUST_CSR_SPMV = not previous
        cortical_column._ensure_native_backends()
        assert cortical_column._HAS_RUST_CSR_SPMV is (not previous)
    finally:
        cortical_column._HAS_RUST_CSR_SPMV = previous


def test_the_deferred_julia_ping_kernel_is_loaded_at_most_once() -> None:
    """The gamma circuit's kernel load is idempotent for the same reason."""
    from sc_neurocore.network import gamma_oscillation

    gamma_oscillation._ensure_julia_ping_step()
    assert gamma_oscillation._JULIA_PING_LOADED is True

    previous = gamma_oscillation._HAS_JULIA_PING_STEP
    try:
        gamma_oscillation._HAS_JULIA_PING_STEP = not previous
        gamma_oscillation._ensure_julia_ping_step()
        assert gamma_oscillation._HAS_JULIA_PING_STEP is (not previous)
    finally:
        gamma_oscillation._HAS_JULIA_PING_STEP = previous
