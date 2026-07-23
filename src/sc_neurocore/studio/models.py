# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Model browser for Studio runtime catalogue entries

"""Model catalogue, descriptor, and simulation helpers for Studio.

Public import path remains ``sc_neurocore.studio.models``; implementation is
split by responsibility into ``model_catalogue``, ``model_simulate``, and
``model_introspection``.
"""

from __future__ import annotations

from sc_neurocore.studio.model_catalogue import (
    ModelMetadataError,
    _introspected_summary,
    get_model_detail,
    list_models,
    model_documentation,
    model_facets,
)
from sc_neurocore.studio.model_introspection import _load_class
from sc_neurocore.studio.model_simulate import (
    RustStudioBackendError,
    RustStudioBackendUnavailable,
    _load_rust_batch_simulate,
    _try_rust_simulate,
    simulate_model,
)

__all__ = [
    "ModelMetadataError",
    "RustStudioBackendError",
    "RustStudioBackendUnavailable",
    "_introspected_summary",
    "_load_class",
    "_load_rust_batch_simulate",
    "_try_rust_simulate",
    "get_model_detail",
    "list_models",
    "model_documentation",
    "model_facets",
    "simulate_model",
]
