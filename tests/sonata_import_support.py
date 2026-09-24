# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_sonata_import.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
from sc_neurocore.adapters.sonata import (
    import_sonata,
    import_sonata_edges,
    import_sonata_nodes,
)


def _stamp(handle: Any) -> None:
    """Mark an HDF5 file as SONATA, as the specification requires."""
    handle.attrs["magic"] = np.uint32(0x0A7A)
    handle.attrs["version"] = np.array([0, 1], dtype=np.uint32)


def _create_nodes_h5(
    path: Path,
    n: int = 10,
    pop_name: str = "exc",
    *,
    properties: dict[str, Any] | None = None,
) -> Path:
    """Create a SONATA nodes file with one population in property group 0."""
    with h5py.File(path, "w") as f:
        _stamp(f)
        grp = f.create_group(f"nodes/{pop_name}")
        grp.create_dataset("node_type_id", data=np.full(n, 100, dtype=np.int64))
        grp.create_dataset("node_group_id", data=np.zeros(n, dtype=np.uint32))
        grp.create_dataset("node_group_index", data=np.arange(n, dtype=np.uint64))
        group = grp.create_group("0")
        for key, values in (properties or {}).items():
            group.create_dataset(key, data=values)
    return path


def _create_edges_h5(
    path: Path,
    src_ids: list[int],
    tgt_ids: list[int],
    weights: list[float] | None = None,
    pop_name: str = "exc_exc",
    *,
    source: str = "exc",
    target: str = "exc",
    delays: list[float] | None = None,
) -> Path:
    """Create a SONATA edges file with one population in property group 0."""
    with h5py.File(path, "w") as f:
        _stamp(f)
        grp = f.create_group(f"edges/{pop_name}")
        sources = grp.create_dataset("source_node_id", data=np.array(src_ids, dtype=np.uint64))
        sources.attrs["node_population"] = source
        targets = grp.create_dataset("target_node_id", data=np.array(tgt_ids, dtype=np.uint64))
        targets.attrs["node_population"] = target
        grp.create_dataset("edge_type_id", data=np.zeros(len(src_ids), dtype=np.int64))
        grp.create_dataset("edge_group_id", data=np.zeros(len(src_ids), dtype=np.uint32))
        grp.create_dataset("edge_group_index", data=np.arange(len(src_ids), dtype=np.uint64))
        g0 = grp.create_group("0")
        if weights is not None:
            g0.create_dataset("syn_weight", data=np.array(weights))
        if delays is not None:
            g0.create_dataset("delay", data=np.array(delays))
    return path


__all__ = [
    "Path",
    "np",
    "pytest",
    "h5py",
    "import_sonata",
    "import_sonata_edges",
    "import_sonata_nodes",
    "_stamp",
    "_create_nodes_h5",
    "_create_edges_h5",
]
