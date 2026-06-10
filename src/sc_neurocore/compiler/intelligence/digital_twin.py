# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Digital twin facade

"""Analog drift compensation and software shadow twin facade."""

from __future__ import annotations

from .drift_compensation import (
    DriftCompensator,
    generate_drift_compensator,
)
from .hil_calibration import (
    HILCalibration,
    generate_hil_calibration,
)
from .seu_scrub_scheduler import (
    ScrubSchedule,
    schedule_seu_scrubbing,
)
from .shadow_twin import (
    generate_digital_twin,
)
from .telemetry_ingestion import (
    TelemetryResult,
    ingest_telemetry,
)

__all__ = [
    "DriftCompensator",
    "HILCalibration",
    "ScrubSchedule",
    "TelemetryResult",
    "generate_digital_twin",
    "generate_drift_compensator",
    "generate_hil_calibration",
    "ingest_telemetry",
    "schedule_seu_scrubbing",
]
