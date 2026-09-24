# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio storage mode selection

"""Keep requested storage isolation separate from HTTP deployment policy."""

from typing import Literal

StudioStorageMode = Literal["embedded", "isolated"]


def parse_storage_mode(value: str | None) -> StudioStorageMode:
    """Parse the explicit environment selection without an unknown-value fallback.

    An absent value preserves embedded compatibility. Empty, misspelled and
    differently cased values raise ``ValueError``; whitespace is stripped.
    Parsing ``isolated`` expresses intent, not availability or qualification.

    Parameters
    ----------
    value:
        Environment value, or ``None`` when the option is absent.

    Returns
    -------
    StudioStorageMode
        Validated selection, without creating storage or changing permissions.

    Raises
    ------
    ValueError
        The supplied value is empty or not a supported mode name.
    """
    if value is None:
        return "embedded"
    selected = value.strip()
    if selected == "embedded":
        return "embedded"
    if selected == "isolated":
        return "isolated"
    raise ValueError("Studio storage mode must be embedded or isolated.")


def require_available_storage(mode: StudioStorageMode) -> None:
    """Refuse an unknown storage selection before any runtime collaborator exists.

    Embedded storage preserves the existing non-isolated filesystem contract.
    Isolated startup additionally requires every check of
    :func:`storage_preflight.require_isolated_preflight` to pass before any
    collaborator is created; it never falls back to a local ledger.

    Parameters
    ----------
    mode:
        Validated storage selection from runtime settings.

    Raises
    ------
    ValueError
        The caller supplied an unknown mode despite the typed contract.
    """
    if mode not in ("embedded", "isolated"):
        raise ValueError("Studio storage mode must be embedded or isolated.")
