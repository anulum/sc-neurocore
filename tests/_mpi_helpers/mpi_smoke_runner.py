# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — MPI smoke-run worker invoked via `mpirun -n N python this_file <out_dir>`

"""Worker invoked under mpirun for the multi-rank MPIRunner integration test.

Builds a deterministic 2-population network, partitions it across the
ranks supplied by the launcher, runs `MPIRunner.run(n_steps=50)`, and
dumps a per-rank JSON to `<out_dir>/rank_<r>.json` describing what the
rank saw.

Designed to be CI-cheap (~1 s) and assertion-friendly: the test driver
checks the JSON dumps for completion, monitor counts, and partition
membership.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from mpi4py import MPI

from sc_neurocore.network import (
    Network,
    Population,
    Projection,
    SpikeMonitor,
)
from sc_neurocore.network.mpi_runner import MPIRunner

# Sanity-touch the real MPI world before we hit MPIRunner. Without
# this, a missing-mpi4py environment would only fail deep inside
# MPIRunner.__init__; this gives a cleaner error.
assert MPI.COMM_WORLD.Get_size() >= 1


def build_network() -> Network:
    """Construct the same deterministic network on every rank.

    Identical seed across ranks is required so that, after partition
    + Allgatherv, the global state is what every rank thinks it should
    be. Rank-local RNG offset (`seed + rank`) is applied inside
    `MPIRunner.run`, so the build itself stays deterministic.
    """
    pop_a = Population("LapicqueNeuron", n=8, label="A")
    pop_b = Population("LapicqueNeuron", n=6, label="B")
    proj_ab = Projection(pop_a, pop_b, weight=1.0, probability=0.5)
    mon_a = SpikeMonitor(pop_a, label="A_spikes")
    mon_b = SpikeMonitor(pop_b, label="B_spikes")
    return Network(pop_a, pop_b, proj_ab, mon_a, mon_b, seed=42)


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    net = build_network()
    runner = MPIRunner(net)

    # Deliberately small timestep count so the test stays under 1 s.
    runner.run(n_steps=50, dt=0.001)

    rank_record: dict[str, object] = {
        "rank": runner.rank,
        "size": runner.size,
        "local_indices": list(runner._local_indices),
        "rank_of": {str(k): v for k, v in runner._rank_of.items()},
        "n_local_projs": len(runner._local_projs),
        "n_cross_rank_projs": len(runner._cross_rank_projs),
        "monitor_counts": [int(m.count) for m in net.spike_monitors],
        "monitor_labels": [m.label for m in net.spike_monitors],
    }

    # Per-rank JSON; the test driver reads them all.
    out_path = out_dir / f"rank_{runner.rank}.json"
    with out_path.open("w") as fh:
        json.dump(rank_record, fh, indent=2, default=_json_default)

    # Coordinate exit: ensure every rank has written before any rank
    # returns. This avoids races when the test driver scans the dir.
    runner.comm.Barrier()


def _json_default(obj: object) -> object:
    """Serialise NumPy scalars/arrays for the JSON dump."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON-serialisable: {type(obj).__name__}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("usage: mpi_smoke_runner.py <out_dir>\n")
        sys.exit(2)
    main(Path(sys.argv[1]))
