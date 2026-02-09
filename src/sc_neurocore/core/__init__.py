"""sc_neurocore.core -- Tier: research (experimental / research)."""

__tier__ = "research"

from .immortality import DigitalSoul
from .mdl_parser import MindDescriptionLanguage, MDLSpecification
from .orchestrator import CognitiveOrchestrator
from .replication import VonNeumannProbe
from .self_awareness import MetaCognitionLoop, SelfModel
from .tensor_stream import TensorStream

__all__ = [
    "DigitalSoul",
    "MindDescriptionLanguage",
    "MDLSpecification",
    "CognitiveOrchestrator",
    "VonNeumannProbe",
    "MetaCognitionLoop",
    "SelfModel",
    "TensorStream",
]
