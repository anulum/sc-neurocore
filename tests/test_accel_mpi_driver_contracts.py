# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for MPI driver contracts

"""Contracts for MPI driver fallback and communicator delegation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


def test_mpi_driver_scatter_and_gather_single_element_fallbacks() -> None:
    """Single-process MPI fallback returns NumPy arrays without mutation."""
    from sc_neurocore.accel.mpi_driver import MPIDriver

    driver = MPIDriver()

    assert isinstance(driver.scatter_workload(np.array([42.0])), np.ndarray)
    assert isinstance(driver.gather_results(np.array([42.0])), np.ndarray)


def test_mpi_driver_import_fallback_uses_single_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing MPI support keeps the driver in deterministic single-rank mode."""
    import sc_neurocore.accel.mpi_driver as module
    from sc_neurocore.accel.mpi_driver import MPIDriver

    monkeypatch.setattr(module, "HAS_MPI", False)
    monkeypatch.setattr(module, "MPI", None)

    driver = MPIDriver()
    src = np.array([10, 11, 12, 13], dtype=np.int32)

    assert driver.comm is None
    assert driver.rank == 0
    assert driver.size == 1
    assert np.array_equal(driver.scatter_workload(src), src)
    assert np.array_equal(driver.gather_results(src), src)


def test_mpi_driver_gather_non_root_returns_typed_empty_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-root MPI ranks return an empty array preserving local result dtype."""
    import sc_neurocore.accel.mpi_driver as module
    from sc_neurocore.accel.mpi_driver import MPIDriver

    class FakeComm:
        """MPI gather stub for non-root ranks."""

        def Gather(
            self,
            local_results: np.ndarray[Any, Any],
            global_results: np.ndarray[Any, Any] | None,
            root: int = 0,
        ) -> None:
            """Validate that non-root ranks do not allocate the receive buffer."""
            assert root == 0
            assert local_results.dtype == np.dtype(np.int32)
            assert global_results is None

    monkeypatch.setattr(module, "HAS_MPI", True)
    driver = MPIDriver()
    driver.size = 2
    driver.rank = 1
    driver.comm = FakeComm()

    out = driver.gather_results(np.array([1, 2, 3], dtype=np.int32))

    assert out.size == 0
    assert out.dtype == np.dtype(np.int32)


def test_mpi_driver_gather_root_returns_collected_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Root MPI rank returns the communicator-filled gather buffer."""
    import sc_neurocore.accel.mpi_driver as module
    from sc_neurocore.accel.mpi_driver import MPIDriver

    class FakeComm:
        """MPI gather stub for the root rank."""

        def Gather(
            self,
            local_results: np.ndarray[Any, Any],
            global_results: np.ndarray[Any, Any] | None,
            root: int = 0,
        ) -> None:
            """Populate the root receive buffer as MPI would."""
            assert root == 0
            assert global_results is not None
            global_results[:] = np.concatenate(
                [local_results, local_results + np.array([2, 2], dtype=np.int32)]
            )

    monkeypatch.setattr(module, "HAS_MPI", True)
    driver = MPIDriver()
    driver.size = 2
    driver.rank = 0
    driver.comm = FakeComm()

    out = driver.gather_results(np.array([1, 2], dtype=np.int32))

    assert np.array_equal(out, np.array([1, 2, 3, 4], dtype=np.int32))


def test_mpi_driver_missing_comm_falls_back_to_local_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing communicator leaves scatter/gather on local arrays."""
    import sc_neurocore.accel.mpi_driver as module
    from sc_neurocore.accel.mpi_driver import MPIDriver

    monkeypatch.setattr(module, "HAS_MPI", True)
    driver = MPIDriver()
    driver.size = 2
    driver.comm = None
    src = np.array([10, 11, 12, 13], dtype=np.int32)

    assert np.array_equal(driver.scatter_workload(src), src)
    assert np.array_equal(driver.gather_results(src), src)


def test_mpi_driver_delegates_barrier_to_comm_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MPI barrier delegates to the active communicator when present."""
    import sc_neurocore.accel.mpi_driver as module
    from sc_neurocore.accel.mpi_driver import MPIDriver

    class FakeComm:
        """MPI barrier stub."""

        def __init__(self) -> None:
            """Initialise the barrier call flag."""
            self.called = False

        def Barrier(self) -> None:
            """Record that the barrier was invoked."""
            self.called = True

    monkeypatch.setattr(module, "HAS_MPI", True)
    driver = MPIDriver()
    fake = FakeComm()
    driver.comm = fake

    driver.barrier()

    assert fake.called is True
