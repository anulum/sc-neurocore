# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Batch model scanning for behavior classification

from __future__ import annotations

from typing import Any
import warnings

from sc_neurocore.studio.codegen import classify_firing_pattern
from sc_neurocore.studio.models import list_models, simulate_model

_CACHE: dict[str, dict[str, Any]] | None = None


def scan_all_models(current: float = 10.0, duration: float = 100.0) -> list[dict[str, Any]]:
    """Simulate every model at a given current and classify its firing pattern.

    Results are cached after first call.
    """
    global _CACHE
    if _CACHE is not None:
        return list(_CACHE.values())

    results: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    models = list_models()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for m in models:
            try:
                r = simulate_model(m["name"], duration=duration, current=current)
                pattern = classify_firing_pattern(r["spikes"], r["n_steps"], r["dt"])
                results[m["name"]] = {
                    "name": m["name"],
                    "category": m.get("category", "Other"),
                    "pattern": pattern["pattern"],
                    "description": pattern["description"],
                    "rate_hz": pattern.get("rate_hz", 0),
                    "spike_count": r["spike_count"],
                }
            except Exception as exc:
                failures.append(
                    {
                        "name": str(m.get("name", "")),
                        "category": str(m.get("category", "Other")),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )

    if failures:
        total = len(models)
        raise ValueError(
            f"model scan failed for {len(failures)}/{total} models",
            {
                "failed_models": failures,
                "failed_count": len(failures),
                "total_models": total,
                "failure_rate": float(len(failures)) / float(max(total, 1)),
            },
        )

    _CACHE = results
    return list(results.values())
