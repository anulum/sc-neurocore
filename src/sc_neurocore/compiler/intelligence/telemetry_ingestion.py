# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Telemetry ingestion

"""Hardware telemetry ingestion and drift comparison logic."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TelemetryResult:
    """Hardware telemetry comparison result.

    Attributes
    ----------
    samples : int
    max_drift : float
    mean_drift : float
    alerts : list[str]
    healthy : bool
    """

    samples: int
    max_drift: float
    mean_drift: float
    alerts: list[str]
    healthy: bool


def ingest_telemetry(
    telemetry_data: list[dict[str, float]],
    twin_states: list[dict[str, float]],
    *,
    drift_threshold: float = 0.1,
) -> TelemetryResult:
    """Ingest hardware telemetry and compare against digital twin."""
    if not telemetry_data or not twin_states:
        return TelemetryResult(
            samples=0,
            max_drift=0.0,
            mean_drift=0.0,
            alerts=[],
            healthy=True,
        )

    n = min(len(telemetry_data), len(twin_states))
    drifts = []
    alerts = []

    for i in range(n):
        hw = telemetry_data[i]
        tw = twin_states[i]
        for var in hw:
            d = abs(hw[var] - tw.get(var, 0.0))
            drifts.append(d)
            if d > drift_threshold:
                alerts.append(f"Sample {i}, var '{var}': drift={d:.4f} > {drift_threshold}")

    max_d = max(drifts) if drifts else 0.0
    mean_d = sum(drifts) / len(drifts) if drifts else 0.0

    return TelemetryResult(
        samples=n,
        max_drift=round(max_d, 6),
        mean_drift=round(mean_d, 6),
        alerts=alerts,
        healthy=len(alerts) == 0,
    )
