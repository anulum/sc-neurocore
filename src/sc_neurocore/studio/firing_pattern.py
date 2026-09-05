# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio firing-pattern classification

"""Name the firing pattern a spike train shows.

Classification describes an observed run and never feeds back into one. It
lives apart from code export because the two answer different questions: what
did this run do, and how does someone else run it again.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Interspike-interval coefficient of variation below which firing is called
#: tonic, and below the second value irregular rather than chaotic.
TONIC_CV = 0.15
IRREGULAR_CV = 0.5

#: A burst is called when the long intervals exceed the short ones by this
#: factor, with "short" and "long" taken relative to the median interval.
BURST_RATIO = 3.0
SHORT_ISI_FRACTION = 0.5
LONG_ISI_FRACTION = 1.5

#: Spike-frequency adaptation is called when the final third of the intervals
#: is at least this much longer than the first third.
ADAPTATION_RATIO = 1.3


def classify_firing_pattern(
    spikes: list[int],
    n_steps: int,
    dt: float,
) -> dict[str, Any]:
    """Classify the firing pattern of a spike train.

    Parameters
    ----------
    spikes : list of int
        Step indices at which the run spiked, in ascending order.
    n_steps : int
        Number of steps the run executed; with ``dt`` it gives the duration
        the rate is computed over.
    dt : float
        Time step in milliseconds.

    Returns
    -------
    dict
        ``pattern`` (``silent``, ``single_spike``, ``bursting``, ``adapting``,
        ``tonic``, ``irregular`` or ``chaotic``) and a human ``description``.
        Every pattern except ``silent`` also carries ``rate_hz``; every pattern
        with at least three spikes carries ``isi_cv``; a burst additionally
        carries ``burst_isi_ms`` and ``inter_burst_ms``.
    """
    if len(spikes) == 0:
        return {"pattern": "silent", "description": "No spikes detected"}

    duration_s = n_steps * dt / 1000.0
    rate = len(spikes) / duration_s if duration_s > 0 else 0

    if len(spikes) < 3:
        return {
            "pattern": "single_spike",
            "description": f"Only {len(spikes)} spike(s)",
            "rate_hz": round(rate, 1),
        }

    isis = np.diff(spikes).astype(float) * dt
    isi_mean = float(np.mean(isis))
    isi_cv = float(np.std(isis) / isi_mean) if isi_mean > 0 else 0

    # Bimodal intervals: short within a burst, long between bursts.
    if len(isis) >= 4:
        median_isi = float(np.median(isis))
        short = isis[isis < median_isi * SHORT_ISI_FRACTION]
        long = isis[isis > median_isi * LONG_ISI_FRACTION]
        if len(short) > 1 and len(long) > 0:
            ratio = float(np.mean(long)) / float(np.mean(short)) if np.mean(short) > 0 else 1
            if ratio > BURST_RATIO:
                return {
                    "pattern": "bursting",
                    "description": f"Burst-pause pattern (ISI ratio {ratio:.1f}x)",
                    "rate_hz": round(rate, 1),
                    "isi_cv": round(isi_cv, 3),
                    "burst_isi_ms": round(float(np.mean(short)), 2),
                    "inter_burst_ms": round(float(np.mean(long)), 2),
                }

    # Adaptation: intervals lengthen over the run.
    if len(isis) >= 5:
        first_third = np.mean(isis[: len(isis) // 3])
        last_third = np.mean(isis[-len(isis) // 3 :])
        if last_third > first_third * ADAPTATION_RATIO:
            return {
                "pattern": "adapting",
                "description": f"Spike-frequency adaptation ({first_third:.1f}→{last_third:.1f} ms ISI)",
                "rate_hz": round(rate, 1),
                "isi_cv": round(isi_cv, 3),
            }

    if isi_cv < TONIC_CV:
        return {
            "pattern": "tonic",
            "description": f"Regular tonic firing (CV={isi_cv:.3f})",
            "rate_hz": round(rate, 1),
            "isi_cv": round(isi_cv, 3),
        }

    if isi_cv < IRREGULAR_CV:
        return {
            "pattern": "irregular",
            "description": f"Irregular spiking (CV={isi_cv:.3f})",
            "rate_hz": round(rate, 1),
            "isi_cv": round(isi_cv, 3),
        }

    return {
        "pattern": "chaotic",
        "description": f"Highly irregular/chaotic (CV={isi_cv:.3f})",
        "rate_hz": round(rate, 1),
        "isi_cv": round(isi_cv, 3),
    }


__all__ = [
    "ADAPTATION_RATIO",
    "BURST_RATIO",
    "IRREGULAR_CV",
    "LONG_ISI_FRACTION",
    "SHORT_ISI_FRACTION",
    "TONIC_CV",
    "classify_firing_pattern",
]
