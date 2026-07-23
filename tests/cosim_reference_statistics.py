# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared co-simulation reference statistics

"""Reduce reference-model trajectories to the common fidelity feature contract."""

from __future__ import annotations

import math


def _summarise(recorded: dict[str, list[float]], spikes: list[int]) -> dict[str, float]:
    """Return the shared spike-count / first-spike-step / per-variable feature map.

    Every reference helper that tracks a per-step ``spikes`` list and one or more
    recorded state-variable trajectories reduces them to the same feature contract: a
    total spike count, the 1-indexed first-spike step (``-1`` when silent), and the
    final / minimum / maximum / mean of each recorded variable. Centralising the tail
    keeps the independent-parity helpers byte-identical in how they summarise, so a
    drift in one helper's reduction cannot silently diverge from the others.

    Parameters
    ----------
    recorded:
        Mapping from state-variable name to its per-step trajectory.
    spikes:
        Per-step spike indicators (``1`` on a spiking step, ``0`` otherwise).

    Returns
    -------
    dict of str to float
        The feature map keyed by ``spike_count``, ``first_spike_step``, and
        ``final.<var>`` / ``min.<var>`` / ``max.<var>`` / ``mean.<var>`` per variable.
    """
    features: dict[str, float] = {
        "spike_count": float(math.fsum(spikes)),
        "first_spike_step": float(
            next((index for index, spike in enumerate(spikes, start=1) if spike), -1)
        ),
    }
    for variable, values in recorded.items():
        features[f"final.{variable}"] = values[-1]
        features[f"min.{variable}"] = min(values)
        features[f"max.{variable}"] = max(values)
        features[f"mean.{variable}"] = math.fsum(values) / len(values)
    return features
