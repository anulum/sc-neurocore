# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Whether a synthesised design fits its target device

"""Judge whether a synthesised design fits its target device, and say why not.

:func:`capacity_verdict` compares a design's counted resources with the
capacity of the device a target is judged against
(:func:`capacity_device_name`); :func:`capacity_sentence` states in words what
the design needs beyond it. Running the tools that count the resources stays in
:mod:`sc_neurocore.studio.synthesis`, and the registered devices and their
capacities in :mod:`sc_neurocore.studio.synthesis_profiles`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sc_neurocore.studio.synthesis_profiles import _CAPACITY_DEVICE

_CAPACITY_KEYS = ("luts", "ffs", "brams", "dsps")


def capacity_verdict(
    resources: Mapping[str, Any],
    capacity: Mapping[str, int],
    *,
    device: str | None = None,
    uncounted: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Say whether a synthesised design fits its target device.

    Synthesis succeeding says the netlist exists, not that the device can
    hold it: a 20-neuron network synthesised to 6237 LUTs for a 5280-LUT
    UP5K and the pipeline reported it complete. Only resources the target's
    capacity lists are judged; a target without capacity data gets no
    verdict rather than a guessed one. Fitting by count is necessary, not
    sufficient: on iCE40 and ECP5 a LUT and a flip-flop share a logic cell
    (708 LUTs and 98 flip-flops took 752 UP5K cells), and placement and
    routing can still fail.

    Parameters
    ----------
    resources : Mapping[str, Any]
        Counted resources of the design.
    capacity : Mapping[str, int]
        The device's capacity per resource.
    device : str or None, optional
        The device the capacity describes, named with the verdict.
    uncounted : Mapping[str, int] or None, optional
        Cells whose cost is unknown, by type. With any, the counts are a
        floor: they can prove a design does not fit, never that it does.

    Returns
    -------
    dict[str, Any]
        ``fits_device`` (True, False, or None when unknown cells leave it
        open), ``exceeds_capacity`` (per resource, what the design needs and
        the device has), ``capacity_device`` and ``uncounted_cells``; an empty
        dict without capacity data.
    """
    if not capacity:
        return {}
    exceeds = {
        key: {"needed": int(resources.get(key, 0)), "available": int(capacity[key])}
        for key in _CAPACITY_KEYS
        if key in capacity and int(resources.get(key, 0)) > int(capacity[key])
    }
    unknown = dict(uncounted or {})
    fits: bool | None = False if exceeds else (None if unknown else True)
    return {
        "fits_device": fits,
        "exceeds_capacity": exceeds,
        "capacity_device": device,
        "uncounted_cells": unknown,
    }


def capacity_device_name(target: str) -> str:
    """Name the device a target's capacity is judged against.

    Parameters
    ----------
    target : str
        Target identifier.

    Returns
    -------
    str
        The device, or the target in capitals when none is named.
    """
    return _CAPACITY_DEVICE.get(target, target.upper())


def capacity_sentence(target: str, exceeds: Mapping[str, Mapping[str, int]]) -> str:
    """Say in words which resources a design needs beyond its device.

    Parameters
    ----------
    target : str
        Target identifier; its judged device is named.
    exceeds : Mapping[str, Mapping[str, int]]
        From :func:`capacity_verdict`.

    Returns
    -------
    str
        One sentence.
    """
    names = {"luts": "LUTs", "ffs": "flip-flops", "brams": "block RAMs", "dsps": "DSP blocks"}
    where = capacity_device_name(target)
    parts = [
        f"{row['needed']} {names.get(key, key)} (the device has {row['available']})"
        for key, row in exceeds.items()
    ]
    return f"the design needs {', '.join(parts)}: it does not fit the {where}"


__all__ = [
    "capacity_device_name",
    "capacity_sentence",
    "capacity_verdict",
]
