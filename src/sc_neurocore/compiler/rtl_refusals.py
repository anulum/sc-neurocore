# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Refusals of the RTL compile, written for the caller

"""Refusals of the RTL compile, written for the caller.

The RTL of an equation neuron has one input port and registers only for its state,
parameters and constants. An equation the RTL cannot bind is refused here, in words
that name the caller's input, rather than emitted as RTL that does not elaborate or
failed with an unmarked fault.
"""

from __future__ import annotations

from collections.abc import Mapping

from sc_neurocore.refusals import AuthoredRefusal


class RtlCompileRefusal(AuthoredRefusal):
    """An equation the RTL compile cannot bind or emit, refused with text for the caller.

    It is a ``ValueError``, as the faults it replaces were.
    """


def reset_target_signal(state_signals: Mapping[str, str], target: str) -> str:
    """Return the register a reset assigns, or refuse a target that is not a state.

    Parameters
    ----------
    state_signals : Mapping[str, str]
        Each state variable's RTL signal name.
    target : str
        The name the reset assigns.

    Returns
    -------
    str
        The target's signal name.

    Raises
    ------
    RtlCompileRefusal
        If the neuron holds no state register for the target.
    """
    if target not in state_signals:
        raise RtlCompileRefusal(
            f"the reset assigns {target!r}, which is not a state of this neuron"
        )
    return state_signals[target]


__all__ = ["RtlCompileRefusal", "reset_target_signal"]
