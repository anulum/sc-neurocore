# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Desktop application startup contract

"""Run the existing Studio backend and interface in an owned desktop window."""

from sc_neurocore.refusals import AuthoredRefusal


class DesktopStartRefused(AuthoredRefusal):
    """A deliberate refusal to start or reuse an unusable desktop profile."""
