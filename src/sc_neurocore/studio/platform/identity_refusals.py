# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Caller-facing Studio identity refusals

"""Distinguish deliberate identity refusals from generated exception text."""

from sc_neurocore.refusals import AuthoredRefusal


class StudioIdentityRefused(AuthoredRefusal):
    """A deliberate identity validation message suitable for callers."""


class StudioIdentityConflict(StudioIdentityRefused):
    """An identity mutation conflicts with an existing persistent identity."""
