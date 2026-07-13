# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hierarchical partitioner backend adapters

"""Flat-buffer encoding and dispatch for maintained KL-refinement kernels."""

from __future__ import annotations

import ctypes
from typing import Any, Protocol, TypeAlias

import numpy as np

from sc_neurocore.chiplet import hierarchical_backend_runtime as runtime
from sc_neurocore.chiplet.hierarchical_graph import CorrelationAwareGraph


Array: TypeAlias = np.ndarray[Any, Any]
EncodedBuffers: TypeAlias = tuple[Array, Array, Array, Array, Array, Array, Array]


class RefinementOwner(Protocol):
    """Structural interface required by backend dispatch."""

    refine_backend: str
    kl_iterations: int
    correlation_penalty: float

    def _refine(
        self,
        partitions: list[list[int]],
        adjacency: dict[int, list[int]],
        graph: CorrelationAwareGraph,
    ) -> list[list[int]]:
        """Run the Python reference refinement."""


def encode_csr(
    partitions: list[list[int]],
    adjacency: dict[int, list[int]],
    graph: CorrelationAwareGraph,
) -> EncodedBuffers:
    """Encode adjacency and ordered partitions into the shared flat ABI."""
    vertex_count = graph.num_vertices
    graph._ensure_edge_cache()
    offsets = np.zeros(vertex_count + 1, dtype=np.int64)
    for vertex in range(vertex_count):
        offsets[vertex + 1] = offsets[vertex] + len(adjacency.get(vertex, []))

    neighbours = np.zeros(int(offsets[-1]), dtype=np.int32)
    scc_abs = np.zeros(int(offsets[-1]), dtype=np.float64)
    for vertex in range(vertex_count):
        base = int(offsets[vertex])
        for index, neighbour in enumerate(adjacency.get(vertex, [])):
            neighbours[base + index] = neighbour
            scc_abs[base + index] = abs(graph.edge_scc(vertex, neighbour))

    vertex_weights = np.asarray(
        [graph.vertex_weights.get(vertex, 1.0) for vertex in range(vertex_count)],
        dtype=np.float64,
    )
    part_map = np.full(vertex_count, -1, dtype=np.int32)
    parts_offsets = np.zeros(len(partitions) + 1, dtype=np.int64)
    for partition_index, partition in enumerate(partitions):
        parts_offsets[partition_index + 1] = parts_offsets[partition_index] + len(partition)
        for vertex in partition:
            part_map[vertex] = partition_index

    parts_concat = np.zeros(int(parts_offsets[-1]), dtype=np.int32)
    for partition_index, partition in enumerate(partitions):
        base = int(parts_offsets[partition_index])
        for index, vertex in enumerate(partition):
            parts_concat[base + index] = vertex
    return (
        offsets,
        neighbours,
        scc_abs,
        vertex_weights,
        part_map,
        parts_concat,
        parts_offsets,
    )


def decode_part_map(part_map: Array, partition_count: int) -> list[list[int]]:
    """Decode a flat vertex-to-partition mapping."""
    partitions: list[list[int]] = [[] for _ in range(partition_count)]
    for vertex, partition in enumerate(part_map):
        partition_index = int(partition)
        if 0 <= partition_index < partition_count:
            partitions[partition_index].append(vertex)
    return partitions


def refine_rust(
    owner: RefinementOwner,
    partitions: list[list[int]],
    adjacency: dict[int, list[int]],
    graph: CorrelationAwareGraph,
) -> list[list[int]]:
    """Run the PyO3 Rust kernel and decode its partition map."""
    kernel = runtime._rust_kl_refine
    if kernel is None:
        raise RuntimeError(
            "Rust KL refine backend requested but py_kl_refine is not "
            "available; install sc_neurocore_engine wheel."
        )
    buffers = encode_csr(partitions, adjacency, graph)
    offsets, neighbours, scc_abs, weights, part_map, concat, part_offsets = buffers
    new_part_map, _moves = kernel(
        offsets,
        neighbours,
        scc_abs,
        weights,
        part_map,
        concat,
        part_offsets,
        len(partitions),
        owner.kl_iterations,
        owner.correlation_penalty,
    )
    return decode_part_map(new_part_map, len(partitions))


def refine_julia(
    owner: RefinementOwner,
    partitions: list[list[int]],
    adjacency: dict[int, list[int]],
    graph: CorrelationAwareGraph,
) -> list[list[int]]:
    """Run the Julia kernel and decode its partition map."""
    kernel = runtime._julia_kl_refine
    if kernel is None:
        raise RuntimeError("Julia KL refine backend not loaded; load the maintained module first.")
    buffers = encode_csr(partitions, adjacency, graph)
    offsets, neighbours, scc_abs, weights, part_map, concat, part_offsets = buffers
    new_part_map = kernel(
        offsets,
        neighbours,
        scc_abs,
        weights,
        part_map.copy(),
        concat,
        part_offsets,
        len(partitions),
        owner.kl_iterations,
        owner.correlation_penalty,
    )
    return decode_part_map(
        np.asarray(new_part_map, dtype=np.int32),
        len(partitions),
    )


def refine_go(
    owner: RefinementOwner,
    partitions: list[list[int]],
    adjacency: dict[int, list[int]],
    graph: CorrelationAwareGraph,
) -> list[list[int]]:
    """Run the typed Go C-shared kernel and decode its partition map."""
    library = runtime._go_kl_refine_lib
    if library is None:
        raise RuntimeError("Go KL refine shared library is not loaded")
    buffers = encode_csr(partitions, adjacency, graph)
    offsets, neighbours, scc_abs, weights, part_map, concat, part_offsets = buffers
    result_map = part_map.copy()
    library.kl_refine_c(
        offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        neighbours.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        scc_abs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        result_map.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        concat.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        part_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(weights.size),
        ctypes.c_int64(scc_abs.size),
        ctypes.c_int32(len(partitions)),
        ctypes.c_int32(owner.kl_iterations),
        ctypes.c_double(owner.correlation_penalty),
    )
    return decode_part_map(result_map, len(partitions))


def refine_mojo(
    owner: RefinementOwner,
    partitions: list[list[int]],
    adjacency: dict[int, list[int]],
    graph: CorrelationAwareGraph,
) -> list[list[int]]:
    """Run the Mojo raw-address kernel and decode its partition map."""
    library = runtime._mojo_kl_refine_lib
    if library is None:
        raise RuntimeError("Mojo KL refine shared library is not loaded")
    buffers = encode_csr(partitions, adjacency, graph)
    offsets, neighbours, scc_abs, weights, part_map, concat, part_offsets = buffers
    result_map = part_map.copy()
    library.kl_refine_c(
        offsets.ctypes.data,
        neighbours.ctypes.data,
        scc_abs.ctypes.data,
        weights.ctypes.data,
        result_map.ctypes.data,
        concat.ctypes.data,
        part_offsets.ctypes.data,
        weights.size,
        scc_abs.size,
        len(partitions),
        owner.kl_iterations,
        owner.correlation_penalty,
    )
    return decode_part_map(result_map, len(partitions))


def dispatch_refine(
    owner: RefinementOwner,
    partitions: list[list[int]],
    adjacency: dict[int, list[int]],
    graph: CorrelationAwareGraph,
) -> list[list[int]]:
    """Dispatch to the requested kernel or the Python reference."""
    backend = owner.refine_backend
    if backend == "rust" and not runtime._HAS_RUST_KL_REFINE:
        raise RuntimeError(
            "Rust KL refine requested but py_kl_refine not available; "
            "install sc_neurocore_engine wheel."
        )
    if backend == "julia" and not runtime._ensure_julia_kl_refine_loaded():
        raise RuntimeError(
            "Julia KL refine requested but juliacall and the maintained module "
            "are unavailable; install juliacall."
        )
    if backend == "go" and not runtime._ensure_go_kl_refine_loaded():
        raise RuntimeError(
            "Go KL refine requested but libpartition.so is not built; run "
            "go build -buildmode=c-shared in accel/go/partition."
        )
    if backend == "mojo" and not runtime._ensure_mojo_kl_refine_loaded():
        raise RuntimeError(
            "Mojo KL refine requested but libpartition.so is not built; run "
            "mojo build --emit shared-lib in accel/mojo/partition."
        )
    if backend == "rust" or (backend == "auto" and runtime._HAS_RUST_KL_REFINE):
        return refine_rust(owner, partitions, adjacency, graph)
    if backend == "julia":
        return refine_julia(owner, partitions, adjacency, graph)
    if backend == "go":
        return refine_go(owner, partitions, adjacency, graph)
    if backend == "mojo":
        return refine_mojo(owner, partitions, adjacency, graph)
    return owner._refine(partitions, adjacency, graph)


__all__: list[str] = []
