"""sc_neurocore.verification -- Tier: research (experimental / research)."""

__tier__ = "research"

from .formal_proofs import FormalVerifier, Interval
from .safety import CodeSafetyVerifier

__all__ = [
    "FormalVerifier",
    "Interval",
    "CodeSafetyVerifier",
]
