# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio server-side NIR -> FPGA Verilog compilation

"""Compile an uploaded NIR graph to synthesisable FPGA Verilog.

This is the server-side counterpart to the Studio's NIR export: a model from any
NIR-emitting framework (snnTorch, Norse, Sinabs, ...) can be written as a standard
``.nir`` (HDF5) file and lowered here to FPGA RTL via the SC-NeuroCore NIR bridge.
The artefacts are returned as plain strings so the API layer can serialise them
without any hardware toolchain installed.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Literal

from sc_neurocore.nir_bridge import compile_network_to_fpga, from_nir, from_scnetwork


def compile_nir_graph(
    graph: Any,
    *,
    module_name: str = "sc_nir_network",
    dt: float = 1.0,
    data_width: int = 16,
    fraction: int = 8,
    source_kind: Literal["lfsr", "sobol"] = "lfsr",
    target: str = "artix7",
) -> dict[str, Any]:
    """Lower a parsed NIR graph to synthesisable Verilog and return the artefacts."""
    network = from_nir(graph, dt=dt)
    neuron_graph = from_scnetwork(network, dt=dt)
    result = compile_network_to_fpga(
        neuron_graph,
        module_name=module_name,
        data_width=data_width,
        fraction=fraction,
        source_kind=source_kind,
        target=target,
    )
    return {
        "module_name": result.module_name,
        "total_neurons": result.total_neurons,
        "total_synapses": result.total_synapses,
        "q_format": result.q_format,
        "top_module": result.top_module,
        "neuron_modules": dict(result.neuron_modules),
        "weight_rom": result.weight_rom,
        "source_modules": dict(result.scnir_source_modules),
        "warnings": list(result.warnings),
    }


def compile_nir_file_bytes(data: bytes, **options: Any) -> dict[str, Any]:
    """Compile a standard ``.nir`` (HDF5) document supplied as raw bytes.

    Options are forwarded to :func:`compile_nir_graph`.
    """
    if not data:
        raise ValueError("Empty NIR upload: no .nir bytes were provided.")
    try:
        import nir
    except ImportError as exc:  # pragma: no cover - exercised only without the nir extra
        raise RuntimeError(
            "NIR compilation requires the 'nir' package: pip install 'sc-neurocore[nir]'"
        ) from exc

    handle, path = tempfile.mkstemp(suffix=".nir")
    os.close(handle)
    try:
        with open(path, "wb") as sink:
            sink.write(data)
        graph = nir.read(path)
    finally:
        os.unlink(path)
    return compile_nir_graph(graph, **options)
