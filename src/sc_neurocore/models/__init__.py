"""sc_neurocore.models -- Tier: research (experimental / research)."""

__tier__ = "research"

from .zoo import SCDigitClassifier, SCKeywordSpotter

__all__ = [
    "SCDigitClassifier",
    "SCKeywordSpotter",
]
