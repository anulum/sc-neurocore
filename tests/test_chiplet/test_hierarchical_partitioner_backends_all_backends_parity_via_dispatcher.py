# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAllBackendsParityViaDispatcher from former test_hierarchical_partitioner_backends.py

"""Focused suite: TestAllBackendsParityViaDispatcher from former test_hierarchical_partitioner_backends.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent))
from hierarchical_partitioner_backends_support import *  # noqa: F403

class TestAllBackendsParityViaDispatcher:
    """Every wired backend (rust/julia/go/mojo) must produce the
    SAME vertex→partition mapping as the Python reference when
    invoked via `HierarchicalPartitioner(refine_backend=...)`.

    These tests exercise the production path (the dispatcher
    invocation), not the bench harness's direct ctypes/juliacall
    calls. They are the load-bearing verification that the wiring
    actually works for callers."""

    @pytest.mark.parametrize(
        "backend,probe_fn,probe_arg",
        [
            ("rust", lambda: __import__("sc_neurocore_engine"), None),
            ("julia", lambda: __import__("juliacall"), None),
            ("go", lambda: None, "go_so"),
            ("mojo", lambda: None, "mojo_so"),
        ],
    )
    def test_dispatcher_backend_matches_python(
        self,
        backend: str,
        probe_fn: Callable[[], object],
        probe_arg: str | None,
    ) -> None:
        # Skip if the backend toolchain or built artefact is missing.
        try:
            probe_fn()
        except ImportError:
            pytest.skip(f"{backend}: prerequisite missing")
        if probe_arg in ("go_so", "mojo_so"):
            from pathlib import Path

            so = Path(__file__).resolve().parents[2] / (
                "src/sc_neurocore/accel/"
                + (
                    "go/partition/libpartition.so"
                    if probe_arg == "go_so"
                    else "mojo/partition/libpartition.so"
                )
            )
            if not so.is_file():
                pytest.skip(f"{backend}: {so.name} not built")

        g = _build_graph(100, avg_degree=8, seed=42)
        hp_py = HierarchicalPartitioner(num_partitions=4, kl_iterations=3, refine_backend="python")
        hp_x = HierarchicalPartitioner(num_partitions=4, kl_iterations=3, refine_backend=backend)
        parts_py, _ = hp_py.partition(g)
        parts_x, _ = hp_x.partition(g)
        pm_py = {v: i for i, p in enumerate(parts_py) for v in p}
        pm_x = {v: i for i, p in enumerate(parts_x) for v in p}
        assert pm_py == pm_x, (
            f"{backend} dispatcher disagrees with Python on "
            f"{sum(1 for v in pm_py if pm_py[v] != pm_x.get(v))} vertex assignments"
        )
