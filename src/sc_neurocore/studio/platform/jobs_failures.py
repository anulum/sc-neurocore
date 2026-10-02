# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Private job diagnostics and public failure messages

"""Keep generated diagnostics in custody and qualify public messages explicitly."""

from __future__ import annotations

from sc_neurocore.refusals import AuthoredRefusal

GENERIC_JOB_FAILURE = "Studio job failed."


class StudioJobError(str):
    """A private diagnostic paired with a deliberately selected public message.

    This is compatible with existing string-valued ledger APIs. Only reviewed
    producers may supply ``public_message``; decoding an untrusted string does
    not confer this qualification. The ledger stores both values separately.

    Parameters
    ----------
    diagnostic : str
        Original private diagnostic retained in the ledger's error column.
    public_message : str
        Deliberately selected caller-facing text, independent of diagnostics.
    """

    __slots__ = ("_public_message",)
    _public_message: str

    def __new__(cls, diagnostic: str, *, public_message: str) -> StudioJobError:
        """Bind retained diagnostic text to a caller-facing message."""
        value = super().__new__(cls, diagnostic)
        value._public_message = public_message
        return value

    @property
    def public_message(self) -> str:
        """Return the message the producer explicitly qualified for callers."""
        return self._public_message

    def __getnewargs_ex__(self) -> tuple[tuple[str], dict[str, str]]:
        """Preserve both messages through existing record pickle transport."""
        return (str(self),), {"public_message": self.public_message}


def authored_job_error(message: str) -> StudioJobError:
    """Qualify a deliberate job supervisor message, never generated fault text.

    Parameters
    ----------
    message : str
        Source-owned lifecycle or policy message, with no generated exception text.

    Returns
    -------
    StudioJobError
        Identical diagnostic and public messages with explicit qualification.
    """
    return StudioJobError(message, public_message=message)


def job_failure(
    error: BaseException,
    *,
    fallback: str = GENERIC_JOB_FAILURE,
    prefix: str = "",
    suffix: str = "",
) -> StudioJobError:
    """Retain the fault; expose its text only when its type marks authored refusal.

    ``fallback``, ``prefix`` and ``suffix`` are source-authored context. Prefix
    and suffix retain diagnostic lifecycle facts without qualifying fault text.

    Parameters
    ----------
    error : BaseException
        Original fault. Its message is public only for AuthoredRefusal.
    fallback : str
        Fixed caller-facing reason when the original type is unmarked.
    prefix : str
        Private diagnostic context prepended to the original fault.
    suffix : str
        Private diagnostic lifecycle or cleanup facts appended to the fault.

    Returns
    -------
    StudioJobError
        Original diagnostic paired with the independently selected public message.
    """
    message = str(error) if isinstance(error, AuthoredRefusal) else fallback
    return StudioJobError(prefix + str(error) + suffix, public_message=message)


def public_job_error(error: str | None) -> str | None:
    """Project a qualified error or a fixed fallback for unqualified diagnostics.

    Parameters
    ----------
    error : str or None
        Retained error; a plain or historical string carries no public provenance.

    Returns
    -------
    str or None
        Explicit public message, the fixed failure fallback, or no error.
    """
    if error is None:
        return None
    return error.public_message if isinstance(error, StudioJobError) else GENERIC_JOB_FAILURE
