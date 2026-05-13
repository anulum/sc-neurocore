# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Intermediate representation utilities

"""Intermediate representation utilities for SC-NeuroCore."""

from __future__ import annotations

from .scnir_schema import (
    SCNIR_PREVIOUS_SCHEMA_VERSION,
    SCNIR_SCHEMA_VERSION,
    SCNIRCorrelationConstraint,
    SCNIRDocument,
    SCNIRPrecision,
    SCNIRSource,
    SCNIRStream,
    SCNIRValidationError,
    load_scnir,
    scnir_from_dict,
    scnir_to_dict,
    upgrade_scnir_dict,
    validate_scnir_dict,
    write_scnir,
)
from .scnir_convert import (
    SCNIRConversionConfig,
    build_scnir_from_neuron_graph,
    export_scnir_from_nir,
)
from .scnir_hdl import (
    SCNIRHDLSourceBundle,
    SCNIRHDLSourceManifestEntry,
    build_scnir_source_bundle,
)

__all__ = [
    "SCNIR_SCHEMA_VERSION",
    "SCNIR_PREVIOUS_SCHEMA_VERSION",
    "SCNIRCorrelationConstraint",
    "SCNIRDocument",
    "SCNIRPrecision",
    "SCNIRSource",
    "SCNIRStream",
    "SCNIRValidationError",
    "SCNIRConversionConfig",
    "SCNIRHDLSourceBundle",
    "SCNIRHDLSourceManifestEntry",
    "build_scnir_from_neuron_graph",
    "build_scnir_source_bundle",
    "export_scnir_from_nir",
    "load_scnir",
    "scnir_from_dict",
    "scnir_to_dict",
    "upgrade_scnir_dict",
    "validate_scnir_dict",
    "write_scnir",
]
