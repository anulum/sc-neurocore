# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NIR (Neuromorphic Intermediate Representation) bridge

"""
NIR integration for SC-NeuroCore.

Provides bidirectional conversion between NIR graphs and SC-NeuroCore
networks.

    >>> import nir
    >>> from sc_neurocore.nir_bridge import from_nir
    >>> graph = nir.read("model.nir")
    >>> network = from_nir(graph, dt=1.0)
    >>> network.run(inputs, steps=100)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .hardware_targets import (
    HardwareNoiseAnnotation,
    NeuromorphicHardwareProfile,
    SCMappingConstraints,
    available_hardware_profiles,
    build_nir_hardware_manifest,
    build_noise_annotation,
    get_hardware_profile,
)
from .silicon_mapping import (
    SiliconMappingConfig,
    build_silicon_mapping_report,
    write_silicon_mapping_report,
)
from .neuromorphic_adapters import (
    NeuromorphicAdapterPackage,
    build_neuromorphic_adapter_bundle,
    build_neuromorphic_adapter_package,
    write_neuromorphic_adapter_bundle,
)
from .neuron_graph import (
    ConnectionSpec,
    NeuronGraph,
    NeuronSpec,
    from_scnetwork,
)
from .quantise_params import (
    QuantisedGraph,
    quantise_graph,
)
from .fpga_compiler import (
    FoldedResourceMetrics,
    NetworkCompilationResult,
    compile_network_to_fpga,
)

_NIR_IMPORT_ERROR: AttributeError | ImportError | None = None
_from_nir_impl: Any | None = None
_to_nir_impl: Any | None = None

try:
    from .parser import SCNetwork as _SCNetwork
    from .parser import from_nir as _from_nir_impl
    from .export import to_nir as _to_nir_impl

    SCNetwork: Any = _SCNetwork
    SCNNetwork: Any = _SCNetwork
except (AttributeError, ImportError) as exc:
    _NIR_IMPORT_ERROR = exc
    SCNetwork = None
    SCNNetwork = None


def from_nir(source: Any, dt: float | None = None, reset_mode: str | None = None) -> Any:
    """Convert a reference NIR graph or file to an executable network.

    Parameters
    ----------
    source : object
        Reference NIR graph or path to its HDF5 file.
    dt : float or None
        Timestep, recorded execution clock, or the foreign-graph default of one.
    reset_mode : str or None
        Reset convention, recorded convention, or the foreign-graph reset default.

    Returns
    -------
    SCNetwork
        Executable graph with independent metadata custody.

    Raises
    ------
    ImportError
        The optional NIR dependency is unavailable.
    ValueError
        The graph, execution metadata or requested options are invalid.
    """
    if _from_nir_impl is None:
        if _NIR_IMPORT_ERROR is None:
            raise ImportError("NIR import failed")
        raise _NIR_IMPORT_ERROR
    return _from_nir_impl(source, dt=dt, reset_mode=reset_mode)


def to_nir(network: Any, path: str | Path | None = None) -> Any:
    """Export a network, original metadata and execution convention to NIR.

    Parameters
    ----------
    network : SCNetwork
        Parsed executable graph. Its dynamic state is not a NIR checkpoint.
    path : str or Path or None
        Optional destination for the reference library's HDF5 writer.

    Returns
    -------
    NIRGraph
        Reference NIR graph, also written to the destination when supplied.

    Raises
    ------
    ImportError
        The optional NIR dependency is unavailable.
    ValueError
        A node or recorded execution contract cannot be exported.
    """
    if _to_nir_impl is None:
        if _NIR_IMPORT_ERROR is None:
            raise ImportError("NIR import failed")
        raise _NIR_IMPORT_ERROR
    return _to_nir_impl(network, path=path)


def from_snntorch(model: Any, sample: Any, *, dt: float = 1e-4, scnir: object | None = None) -> Any:
    """Import a supported CPU float64 snnTorch sequence as fresh-state inference.

    Parameters
    ----------
    model : torch.nn.Sequential
        Flat Linear, Leaky and Synaptic sequence using immediate uniform reset.
    sample : torch.Tensor
        Finite unbatched vector defining the input boundary.
    dt : float
        Positive finite NIR timestep.
    scnir : object or None
        Attached stochastic contracts are explicitly refused.

    Returns
    -------
    SCNetwork
        Independent inference graph. See the snnTorch interchange documentation
        for precision, reset and training boundaries.
    """
    from .snntorch_bridge import from_snntorch as convert

    return convert(model, sample, dt=dt, scnir=scnir)


def to_snntorch(network: Any, *, scnir: object | None = None) -> Any:
    """Export a supported float reference chain as actual snnTorch modules.

    Parameters
    ----------
    network : SCNetwork
        Single-port Affine, Linear, IF, LIF or CubaLIF chain with zero potentials.
    scnir : object or None
        SC bitstream length, encoding and fixed-point contracts are refused.

    Returns
    -------
    torch.nn.Sequential
        Independent CPU float64 modules starting from reset state.
    """
    from .snntorch_bridge import to_snntorch as convert

    return convert(network, scnir=scnir)


__all__ = [
    "from_snntorch",
    "to_snntorch",
    "from_nir",
    "to_nir",
    "SCNetwork",
    "SCNNetwork",
    "ConnectionSpec",
    "NeuronGraph",
    "NeuronSpec",
    "FoldedResourceMetrics",
    "NetworkCompilationResult",
    "QuantisedGraph",
    "compile_network_to_fpga",
    "from_scnetwork",
    "quantise_graph",
    "HardwareNoiseAnnotation",
    "NeuromorphicHardwareProfile",
    "NeuromorphicAdapterPackage",
    "SCMappingConstraints",
    "SiliconMappingConfig",
    "available_hardware_profiles",
    "build_neuromorphic_adapter_bundle",
    "build_neuromorphic_adapter_package",
    "build_nir_hardware_manifest",
    "build_noise_annotation",
    "build_silicon_mapping_report",
    "get_hardware_profile",
    "write_neuromorphic_adapter_bundle",
    "write_silicon_mapping_report",
]
