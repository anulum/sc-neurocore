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


def from_nir(source: Any, dt: float = 1.0, reset_mode: str = "reset") -> Any:
    """Convert a NIR graph/source to an SC-NeuroCore network."""

    if _from_nir_impl is None:
        if _NIR_IMPORT_ERROR is None:
            raise ImportError("NIR import failed")
        raise _NIR_IMPORT_ERROR
    return _from_nir_impl(source, dt=dt, reset_mode=reset_mode)


def to_nir(network: Any, path: str | Path | None = None) -> Any:
    """Export an SC-NeuroCore network to NIR."""

    if _to_nir_impl is None:
        if _NIR_IMPORT_ERROR is None:
            raise ImportError("NIR import failed")
        raise _NIR_IMPORT_ERROR
    return _to_nir_impl(network, path=path)


__all__ = [
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
