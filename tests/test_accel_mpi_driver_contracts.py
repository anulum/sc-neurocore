# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for MPI driver contracts

"""Contracts for MPI driver fallback and communicator delegation."""

from __future__ import annotations

import numpy as np


def test_mpi_driver_scatter_and_gather_single_element_fallbacks() -> None:
    from sc_neurocore.accel.mpi_driver import MPIDriver

    driver = MPIDriver()

    assert isinstance(driver.scatter_workload(np.array([42.0])), np.ndarray)
    assert isinstance(driver.gather_results(np.array([42.0])), np.ndarray)


def test_mpi_driver_gather_non_root_returns_empty_array(monkeypatch) -> None:
    import sc_neurocore.accel.mpi_driver as module
    from sc_neurocore.accel.mpi_driver import MPIDriver

    class FakeComm:
        def Gather(self, local_results, global_results, root=0):
            assert global_results is None

    monkeypatch.setattr(module, "HAS_MPI", True)
    driver = MPIDriver()
    driver.size = 2
    driver.rank = 1
    driver.comm = FakeComm()

    out = driver.gather_results(np.array([1, 2, 3], dtype=np.int32))

    assert out.size == 0


def test_mpi_driver_missing_comm_falls_back_to_local_arrays(monkeypatch) -> None:
    import sc_neurocore.accel.mpi_driver as module
    from sc_neurocore.accel.mpi_driver import MPIDriver

    monkeypatch.setattr(module, "HAS_MPI", True)
    driver = MPIDriver()
    driver.size = 2
    driver.comm = None
    src = np.array([10, 11, 12, 13], dtype=np.int32)

    assert np.array_equal(driver.scatter_workload(src), src)
    assert np.array_equal(driver.gather_results(src), src)


def test_mpi_driver_delegates_barrier_to_comm_when_available(monkeypatch) -> None:
    import sc_neurocore.accel.mpi_driver as module
    from sc_neurocore.accel.mpi_driver import MPIDriver

    class FakeComm:
        def __init__(self):
            self.called = False

        def Barrier(self):
            self.called = True

    monkeypatch.setattr(module, "HAS_MPI", True)
    driver = MPIDriver()
    fake = FakeComm()
    driver.comm = fake

    driver.barrier()

    assert fake.called is True
