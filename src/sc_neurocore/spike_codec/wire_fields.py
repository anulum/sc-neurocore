# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fixed-width fields of the spike codec wire formats

"""Check a value against an unsigned 32-bit field of a spike codec wire format.

The ISI codec's header dimensions and the Huffman coder's bit count are refused by name, with a
``ValueError``, before they are packed, so a caller never sees a bare ``struct.error`` and no
packet is written with a wrapped value. Other ``!I`` fields of the codecs do not pass through
this check yet.

The function is public, not private, because its refusal is tested directly; the repository's
test policy forbids tests through private helpers.
"""

from __future__ import annotations

from operator import index
from typing import SupportsIndex


def uint32_field(value: SupportsIndex, what: str) -> int:
    """Return ``value`` as an ``int`` that fits an unsigned 32-bit field, or refuse it.

    Parameters
    ----------
    value : SupportsIndex
        The value to be written into the field.
    what : str
        What the field holds; the refusal names it.

    Returns
    -------
    int
        The value, in ``[0, 2**32)``.

    Raises
    ------
    ValueError
        If the value is negative or at least ``2**32``; the message is
        ``"<what> must fit uint32"``.
    TypeError
        If the value is not an integer.
    """
    checked = index(value)
    if not 0 <= checked < 1 << 32:
        raise ValueError(f"{what} must fit uint32")
    return checked


__all__ = ["uint32_field"]
