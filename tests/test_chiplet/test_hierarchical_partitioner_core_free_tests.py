# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Module-level tests from former test_hierarchical_partitioner_core.py

"""Module-level tests from former test_hierarchical_partitioner_core.py."""

from __future__ import annotations

from hierarchical_partitioner_core_support import *  # noqa: F403

def test_historical_flat_buffer_wrappers_cover_invalid_partition_ids() -> None:
    """The historical private ABI wrappers retain their filtering contract."""
    graph = _build_graph(4, avg_degree=2, seed=9)
    partitioner = HierarchicalPartitioner(num_partitions=2)
    buffers = partitioner._encode_csr([[0, 2], [1, 3]], graph.adjacency(), graph)
    assert buffers[0].shape == (5,)
    decoded = partitioner._decode_part_map(
        np.asarray([-1, 0, 2, 1], dtype=np.int32),
        2,
    )
    assert decoded == [[1], [3]]
