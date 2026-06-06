# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Exotic physics facade

"""Frontier physics compilation facade (photonic, quantum, wetware)."""

from __future__ import annotations

from .adiabatic_clocks import (
    AdiabaticPhase,
    generate_adiabatic_clocks,
)
from .cognitive_bounds import (
    CognitiveBounds,
    enforce_cognitive_bounds,
)
from .holographic_interconnect import (
    HolographicRouter,
    route_holographic_interconnects,
)
from .morphology_synth import (
    Morphology,
    synthesize_morphology,
)
from .omni_paradigm import (
    OmniDispatchMap,
    dispatch_omni_paradigm,
)
from .optical_encoding import (
    MZIWeightEncoding,
    encode_mzi_weights,
    generate_mzi_config,
)
from .reversible_logic import (
    ReversibleNetlist,
    synthesize_reversible_logic,
)
from .wetware_mea import (
    MEAMapping,
    map_wetware_mea,
)

__all__ = [
    "AdiabaticPhase",
    "CognitiveBounds",
    "HolographicRouter",
    "MEAMapping",
    "MZIWeightEncoding",
    "Morphology",
    "OmniDispatchMap",
    "ReversibleNetlist",
    "dispatch_omni_paradigm",
    "encode_mzi_weights",
    "enforce_cognitive_bounds",
    "generate_adiabatic_clocks",
    "generate_mzi_config",
    "map_wetware_mea",
    "route_holographic_interconnects",
    "synthesize_morphology",
    "synthesize_reversible_logic",
]
