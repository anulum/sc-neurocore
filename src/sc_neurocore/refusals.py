# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Explicit caller-facing refusal messages

"""Mark deliberate domain messages separately from generated exception text."""


class AuthoredRefusal(ValueError):
    """A deliberate caller-facing refusal, compatible with ValueError handlers.

    Raise this type or a domain subclass only with text written for callers.
    Do not wrap generated exception text in it. HTTP boundaries should accept
    their own domain subclass and use a fixed message for every other fault.
    """
