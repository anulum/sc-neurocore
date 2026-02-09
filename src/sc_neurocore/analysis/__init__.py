"""sc_neurocore.analysis -- Tier: research (experimental / research)."""

__tier__ = "research"

from .consciousness import PhiEvaluator
from .explainability import SpikeToConceptMapper
from .kardashev import KardashevEstimator
from .qualia import QualiaTuringTest

__all__ = [
    "PhiEvaluator",
    "SpikeToConceptMapper",
    "KardashevEstimator",
    "QualiaTuringTest",
]
