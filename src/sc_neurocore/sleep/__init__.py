"""
Sleep Optimization Package
===========================

Full closed-loop sleep optimization system:
- Sleep stage detection from EEG band powers
- Circadian profiling (4 chronotypes)
- Protocol library (insomnia, jet lag, deep sleep, REM, shift work, power nap)
- Adaptive audio per sleep stage
- Morning report generation

Modules:
    sleep_stage_detector - EEG → sleep stage classification
    circadian_optimizer  - Chronotype-based circadian profiling
    protocol_library     - Sleep protocol templates
    sleep_optimizer      - Master closed-loop orchestrator
    report_generator     - Morning report with hypnogram + metrics

Author: Claude (Session 2026-02-16)
"""

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
