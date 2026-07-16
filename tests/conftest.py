# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared test configuration and fixtures for SC-NeuroCore

"""
Shared test configuration and fixtures for SC-NeuroCore.
"""

import os
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest
from filelock import FileLock

from tests.reload_guard import restore_first_party_reloads

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CARGO_LIB_LOCK = _REPO_ROOT / "engine" / "target" / ".cargo-lib-test.lock"


def _run_cargo_lib_test(test_filter: str) -> subprocess.CompletedProcess[str]:
    """Run ``cargo test <test_filter> --lib`` serialised by a cross-process lock.

    pytest-xdist distributes the UltraScale+ and DCLS Rust contract tests across
    workers (CI uses ``--dist loadfile``), so without serialisation their
    ``cargo test`` subprocesses can run concurrently against the shared
    ``engine/target`` directory and race during a build, surfacing as ``cargo``
    exit status 101 on the slow coverage job. A file lock keyed on
    ``engine/target`` serialises the invocations without changing the suite-wide
    xdist distribution mode.

    Parameters
    ----------
    test_filter : str
        Substring passed to ``cargo test`` to select the library tests to run.

    Returns
    -------
    subprocess.CompletedProcess[str]
        The completed ``cargo`` process with captured text output.

    Raises
    ------
    subprocess.CalledProcessError
        If ``cargo test`` exits with a non-zero status.
    """
    _CARGO_LIB_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(_CARGO_LIB_LOCK)):
        return subprocess.run(
            ["cargo", "test", test_filter, "--lib"],
            cwd=_REPO_ROOT / "engine",
            check=True,
            capture_output=True,
            text=True,
        )


@pytest.fixture
def cargo_lib_test() -> Callable[[str], subprocess.CompletedProcess[str]]:
    """Provide a lock-serialised ``cargo test <filter> --lib`` runner.

    Returns
    -------
    Callable[[str], subprocess.CompletedProcess[str]]
        Callable taking a ``cargo test`` filter substring and returning the
        completed process; raises ``subprocess.CalledProcessError`` on failure.
    """
    return _run_cargo_lib_test


@pytest.fixture(autouse=True)
def restore_repo_cwd() -> Iterator[None]:
    """Keep process-wide CWD changes from leaking across tests."""
    os.chdir(_REPO_ROOT)
    try:
        yield
    finally:
        os.chdir(_REPO_ROOT)


@pytest.fixture(autouse=True)
def seed_random() -> Iterator[None]:
    """Seed numpy RNG before every test for deterministic results."""
    np.random.seed(42)
    yield


@pytest.fixture(autouse=True)
def restore_first_party_module_identity() -> Iterator[None]:
    """Restore any first-party module reloaded during a test to its original identities.

    ``importlib.reload`` of an ``sc_neurocore`` module rebinds its classes to fresh objects; a
    test that forgets to restore them leaks those identities to whichever later test lands in
    the same worker, which passes under ``-n auto --dist loadfile`` but fails a serial run
    deterministically. This guard makes the whole reload-without-restore class a no-op — see
    :mod:`tests.reload_guard`.
    """
    with restore_first_party_reloads():
        yield
