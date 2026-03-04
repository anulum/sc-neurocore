"""sc_neurocore.sleep -- Tier: research (experimental / research).

Closed-loop sleep optimisation system with EEG-based stage detection,
circadian-aware protocol selection, adaptive audio entrainment, and
post-session quality reporting.
"""

__tier__ = "research"

from .sleep_stage_detector import SleepStageDetector, SleepStage
from .circadian_optimizer import CircadianOptimizer, Chronotype
from .protocol_library import SleepProtocol, get_protocol, list_protocols
from .sleep_optimizer import SleepOptimizer, SleepOptimizerConfig
from .report_generator import SleepReportGenerator, SleepReport

__all__ = [
    "SleepStageDetector",
    "SleepStage",
    "CircadianOptimizer",
    "Chronotype",
    "SleepProtocol",
    "get_protocol",
    "list_protocols",
    "SleepOptimizer",
    "SleepOptimizerConfig",
    "SleepReportGenerator",
    "SleepReport",
]
