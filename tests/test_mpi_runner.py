# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for MPI distributed simulation runner

"""Tests for sc_neurocore.network.mpi_runner."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sc_neurocore.network.mpi_runner import HAS_MPI, _require_mpi


def test_import_without_mpi():
    """MPIRunner module loads even when mpi4py is absent; HAS_MPI reflects availability."""
    assert isinstance(HAS_MPI, bool)


def test_require_mpi_raises_without_mpi():
    """_require_mpi raises RuntimeError when HAS_MPI is False."""
    with (
        patch("sc_neurocore.network.mpi_runner.HAS_MPI", False),
        pytest.raises(RuntimeError, match="mpi4py is required"),
    ):
        _require_mpi()


def _make_network():
    """Build a minimal 2-population network for testing."""
    from sc_neurocore.network import Network, Population, Projection, SpikeMonitor

    pop_a = Population("LapicqueNeuron", 4, label="A")
    pop_b = Population("LapicqueNeuron", 3, label="B")
    proj = Projection(pop_a, pop_b, weight=1.0, probability=0.5)
    mon = SpikeMonitor(pop_a, label="A_spikes")
    return Network(pop_a, pop_b, proj, mon, seed=42)


def test_partition_populations():
    """Round-robin partitioning assigns populations to ranks correctly."""
    net = _make_network()

    comm_mock = MagicMock()
    comm_mock.Get_rank.return_value = 0
    comm_mock.Get_size.return_value = 2

    with (
        patch("sc_neurocore.network.mpi_runner.HAS_MPI", True),
        patch("sc_neurocore.network.mpi_runner.MPI") as mpi_mock,
    ):
        mpi_mock.COMM_WORLD = comm_mock
        from sc_neurocore.network.mpi_runner import MPIRunner

        runner = MPIRunner(net)

    assert runner._local_indices == [0]
    assert runner._rank_of == {0: 0, 1: 1}


def test_single_rank_matches_python():
    """Single-rank MPI simulation produces same spike count as Python backend."""
    net_py = _make_network()
    net_py.run(0.05, dt=0.001, backend="python")
    py_spikes = sum(m.count for m in net_py.spike_monitors)

    net_mpi = _make_network()

    comm_mock = MagicMock()
    comm_mock.Get_rank.return_value = 0
    comm_mock.Get_size.return_value = 1

    def fake_allgather(send, recv):
        recv[:] = send

    def fake_allgatherv(send, recv_info):
        recv_buf = recv_info[0]
        recv_buf[: len(send)] = send

    comm_mock.Allgather = fake_allgather
    comm_mock.Allgatherv = fake_allgatherv

    with (
        patch("sc_neurocore.network.mpi_runner.HAS_MPI", True),
        patch("sc_neurocore.network.mpi_runner.MPI") as mpi_mock,
    ):
        mpi_mock.COMM_WORLD = comm_mock
        mpi_mock.BYTE = 0
        from sc_neurocore.network.mpi_runner import MPIRunner

        runner = MPIRunner(net_mpi)
        runner.run(50, dt=0.001)

    mpi_spikes = sum(m.count for m in net_mpi.spike_monitors)
    assert mpi_spikes == py_spikes


def test_cross_rank_projection_identification():
    """Projections spanning rank boundaries are classified as cross-rank."""
    net = _make_network()

    comm_mock = MagicMock()
    comm_mock.Get_rank.return_value = 0
    comm_mock.Get_size.return_value = 2

    with (
        patch("sc_neurocore.network.mpi_runner.HAS_MPI", True),
        patch("sc_neurocore.network.mpi_runner.MPI") as mpi_mock,
    ):
        mpi_mock.COMM_WORLD = comm_mock
        from sc_neurocore.network.mpi_runner import MPIRunner

        runner = MPIRunner(net)

    # pop_a (idx 0) -> rank 0, pop_b (idx 1) -> rank 1
    # Projection A->B crosses rank boundary
    assert len(runner._cross_rank_projs) == 1
    assert len(runner._local_projs) == 0


def test_exchange_spikes_mock():
    """Spike exchange packs/unpacks correctly in single-rank mode."""
    net = _make_network()

    comm_mock = MagicMock()
    comm_mock.Get_rank.return_value = 0
    comm_mock.Get_size.return_value = 1

    def fake_allgather(send, recv):
        recv[:] = send

    def fake_allgatherv(send, recv_info):
        recv_buf = recv_info[0]
        recv_buf[: len(send)] = send

    comm_mock.Allgather = fake_allgather
    comm_mock.Allgatherv = fake_allgatherv

    with (
        patch("sc_neurocore.network.mpi_runner.HAS_MPI", True),
        patch("sc_neurocore.network.mpi_runner.MPI") as mpi_mock,
    ):
        mpi_mock.COMM_WORLD = comm_mock
        mpi_mock.BYTE = 0
        from sc_neurocore.network.mpi_runner import MPIRunner

        runner = MPIRunner(net)
        spikes = {0: np.array([1, 0, 1, 0], dtype=np.int8), 1: np.array([0, 1, 0], dtype=np.int8)}
        result = runner._exchange_spikes(spikes)

    assert 0 in result
    assert 1 in result
    np.testing.assert_array_equal(result[0], [1, 0, 1, 0])
    np.testing.assert_array_equal(result[1], [0, 1, 0])
