# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — sc_neurocore.world_model -- Tier: research (experimental

"""sc_neurocore.world_model -- Tier: research (experimental / research)."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__tier__ = "research"

from .spike_predictor import SpikePredictor

_LAZY_EXPORTS = {
    "SCPlanner": (".planner", "SCPlanner"),
    "PredictiveWorldModel": (".predictive_model", "PredictiveWorldModel"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_name, symbol_name = _LAZY_EXPORTS[name]
        module = import_module(module_name, __name__)
        value = getattr(module, symbol_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from .planner import SCPlanner
    from .predictive_model import PredictiveWorldModel

__all__ = [
    "SCPlanner",
    "PredictiveWorldModel",
    "SpikePredictor",
]
