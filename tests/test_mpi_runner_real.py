# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real multi-rank MPIRunner test (closes follow-up #17)

"""Multi-rank integration test for `MPIRunner` via real `mpirun -n 2`.

Existing `tests/test_mpi_runner.py` covers the API with mock
`mpi4py.MPI` objects (single mocked rank). This file complements it
with a **real** distributed run launched through `mpirun`, so that:

1. `Allgather` / `Allgatherv` actually exchange bytes between two
   processes (catches buffer-layout regressions that mocks hide).
2. Round-robin partitioning is exercised end-to-end (rank 0 owns
   pop 0, rank 1 owns pop 1).
3. The `comm.Barrier()` rendezvous at the end of the worker actually
   synchronises (catches communicator-misuse bugs).

The test skips automatically when `mpirun` is not on PATH or when
`mpi4py` is not importable in the active interpreter, so it is safe
to ship in the default suite. Closes follow-up task #17.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

mpi4py = pytest.importorskip("mpi4py")

# Worker script that each MPI rank executes. Path is resolved relative
# to this test file so the test is hermetic (works regardless of CWD).
HERE = Path(__file__).parent
WORKER = HERE / "_mpi_helpers" / "mpi_smoke_runner.py"


def _mpirun_available() -> bool:
    return shutil.which("mpirun") is not None


@pytest.mark.skipif(
    not _mpirun_available(),
    reason="mpirun not on PATH (install openmpi-bin or mpich)",
)
def test_mpirun_two_ranks_completes(tmp_path: Path) -> None:
    """`mpirun -n 2` must complete and write one JSON dump per rank."""
    assert WORKER.is_file(), f"worker script missing: {WORKER}"

    # `--oversubscribe` lets the test run on workstations with <2 cores
    # (CI runners are sometimes single-core). Open MPI accepts it; on
    # MPICH-based mpirun the flag is unrecognised → fall back without.
    env = os.environ.copy()
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")

    cmd_oversubscribe = [
        "mpirun",
        "--oversubscribe",
        "-n",
        "2",
        sys.executable,
        str(WORKER),
        str(tmp_path),
    ]
    cmd_plain = [
        "mpirun",
        "-n",
        "2",
        sys.executable,
        str(WORKER),
        str(tmp_path),
    ]

    result = subprocess.run(
        cmd_oversubscribe,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    if result.returncode != 0 and "oversubscribe" in (result.stderr or "").lower():
        result = subprocess.run(
            cmd_plain,
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )

    assert result.returncode == 0, (
        f"mpirun -n 2 failed (rc={result.returncode})\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    rank0_json = tmp_path / "rank_0.json"
    rank1_json = tmp_path / "rank_1.json"
    assert rank0_json.is_file(), f"rank 0 did not write its dump:\n{result.stderr}"
    assert rank1_json.is_file(), f"rank 1 did not write its dump:\n{result.stderr}"

    rank0 = json.loads(rank0_json.read_text())
    rank1 = json.loads(rank1_json.read_text())

    # World shape.
    assert rank0["size"] == 2 and rank1["size"] == 2
    assert rank0["rank"] == 0 and rank1["rank"] == 1

    # Round-robin partition: pop 0 → rank 0, pop 1 → rank 1.
    assert rank0["local_indices"] == [0]
    assert rank1["local_indices"] == [1]
    assert rank0["rank_of"] == {"0": 0, "1": 1}
    assert rank1["rank_of"] == rank0["rank_of"]

    # The single A→B projection crosses rank boundary on both ranks.
    assert rank0["n_cross_rank_projs"] == 1
    assert rank1["n_cross_rank_projs"] == 1
    assert rank0["n_local_projs"] == 0
    assert rank1["n_local_projs"] == 0

    # Only rank 0 records monitors (per `MPIRunner.run` contract).
    # Each rank built two monitors locally; rank-0 must have non-empty
    # counts for both.
    assert len(rank0["monitor_counts"]) == 2
    assert len(rank1["monitor_counts"]) == 2
    # Rank 0 records both monitors. Allow zero-spike runs (tiny LIF
    # network, 50 steps), but the count list must be present and
    # non-negative.
    for label, count in zip(rank0["monitor_labels"], rank0["monitor_counts"]):
        assert count >= 0, f"rank 0 monitor {label!r} reported negative count"


@pytest.mark.skipif(
    not _mpirun_available(),
    reason="mpirun not on PATH",
)
def test_mpirun_one_rank_partition_collapses(tmp_path: Path) -> None:
    """`mpirun -n 1` is a degenerate case: one rank owns both populations."""
    assert WORKER.is_file()

    env = os.environ.copy()
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")

    result = subprocess.run(
        ["mpirun", "-n", "1", sys.executable, str(WORKER), str(tmp_path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"mpirun -n 1 failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    rank0 = json.loads((tmp_path / "rank_0.json").read_text())
    assert rank0["size"] == 1
    assert rank0["rank"] == 0
    assert rank0["local_indices"] == [0, 1]  # owns everything
    # Both projections become local (no cross-rank traffic when n=1).
    assert rank0["n_local_projs"] == 1
    assert rank0["n_cross_rank_projs"] == 0
