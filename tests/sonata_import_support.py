# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_sonata_import.py

from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
from sc_neurocore.adapters.sonata import (
    import_sonata,
    import_sonata_edges,
    import_sonata_nodes,
)


def _create_nodes_h5(path, n=10, pop_name="exc"):
    """Create a minimal SONATA nodes HDF5 file."""
    with h5py.File(path, "w") as f:
        grp = f.create_group(f"nodes/{pop_name}")
        grp.create_dataset("node_id", data=np.arange(n))
        grp.create_dataset("node_type_id", data=np.zeros(n, dtype=int))
    return path


def _create_edges_h5(path, src_ids, tgt_ids, weights=None, pop_name="exc_exc"):
    """Create a minimal SONATA edges HDF5 file."""
    with h5py.File(path, "w") as f:
        grp = f.create_group(f"edges/{pop_name}")
        grp.create_dataset("source_node_id", data=np.array(src_ids))
        grp.create_dataset("target_node_id", data=np.array(tgt_ids))
        grp.create_dataset("edge_type_id", data=np.zeros(len(src_ids), dtype=int))
        if weights is not None:
            g0 = grp.create_group("0")
            g0.create_dataset("syn_weight", data=np.array(weights))
    return path


__all__ = [
    "np",
    "pytest",
    "h5py",
    "import_sonata",
    "import_sonata_edges",
    "import_sonata_nodes",
    "_create_nodes_h5",
    "_create_edges_h5",
]
