# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The bound on one equation compile's work

"""Bound the work of one equation compile before the RTL emitter runs.

The RTL emitter writes a lookup table for every function call, about 11,000
characters of RTL each; a balanced sum of 1,024 ``tanh`` calls emitted 11.3
million characters in 0.48 s. ``/api/compile`` checks a request by compiling it
in the API process before it admits a job, and ``/api/ir/emit-sv-direct``
compiles in that process too, so the work of one request is bounded here, on
the text it sends, before either compiles: at most :data:`MAX_FUNCTION_CALLS`
function calls and :data:`MAX_EXPRESSION_CHARACTERS` characters over its
equations, threshold and reset. At the bound, 128 calls emit 1.4 million
characters in 0.06 s. The largest legitimate source, over every bound catalogue
schema, Studio template and Studio preset, calls 25 functions; the longest
holds 3,204 characters.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from sc_neurocore.refusals import AuthoredRefusal

#: The most function calls one compile may parse, over its equations, threshold and reset.
MAX_FUNCTION_CALLS = 128

#: The most characters one compile's equations, threshold and reset may hold together.
MAX_EXPRESSION_CHARACTERS = 16_384

#: An ``=`` that assigns, not one of ``==``, ``<=``, ``>=`` or ``!=``.
_ASSIGNMENT = re.compile(r"(?<![=!<>])=(?!=)")


class CompileWorkRefused(AuthoredRefusal):
    """An equation compile over the bound on its work, refused with text for the caller."""


@dataclass(frozen=True)
class CompileWork:
    """The work one equation compile asks for.

    Attributes
    ----------
    function_calls : int
        Function calls over every expression the compile parses.
    characters : int
        Characters of the equations, the threshold and the reset together.
    """

    function_calls: int
    characters: int


def _texts(equations: Sequence[str], threshold: str | None, reset: str | None) -> list[str]:
    """Return the request's texts: each equation, then the threshold and the reset when given.

    Parameters
    ----------
    equations : Sequence[str]
        The request's equations.
    threshold : str or None
        The request's threshold.
    reset : str or None
        The request's reset.

    Returns
    -------
    list[str]
        The texts, unchanged.
    """
    return [*equations, *(text for text in (threshold, reset) if text is not None)]


def _parts(texts: Sequence[str]) -> Iterator[str]:
    """Yield each ``;``-separated, non-blank part of the texts, stripped, as the compile splits them.

    Parameters
    ----------
    texts : Sequence[str]
        The request's texts.

    Yields
    ------
    str
        One part.
    """
    for text in texts:
        for part in text.split(";"):
            if part.strip():
                yield part.strip()


def _function_calls(part: str) -> int:
    """Count the function calls in one part, or bound them by its opening parentheses.

    The part is read as Python both ways the compile reads one: whole (a
    threshold, or a reset rule) and the side right of its assigning ``=`` (an
    equation). The larger count of the readings that parse is taken, so neither
    reading can hold more calls than are counted. A part that parses neither way
    counts each ``(`` as a call, since no call is written without one.

    Parameters
    ----------
    part : str
        One equation, threshold or reset rule.

    Returns
    -------
    int
        The number of calls, or the number of ``(`` when the part does not parse.
    """
    counts: list[int] = []
    for candidate in (part, _ASSIGNMENT.split(part, maxsplit=1)[-1].strip()):
        try:
            tree = ast.parse(candidate, mode="exec")
        except SyntaxError:
            continue
        counts.append(sum(isinstance(node, ast.Call) for node in ast.walk(tree)))
    return max(counts) if counts else part.count("(")


def measure_compile_work(
    equations: Sequence[str], threshold: str | None, reset: str | None
) -> CompileWork:
    """Measure the work one equation compile asks for.

    Parameters
    ----------
    equations : Sequence[str]
        The request's equations.
    threshold : str or None
        The request's threshold.
    reset : str or None
        The request's reset.

    Returns
    -------
    CompileWork
        Its function calls and its characters.
    """
    texts = _texts(equations, threshold, reset)
    return CompileWork(
        function_calls=sum(_function_calls(part) for part in _parts(texts)),
        characters=sum(len(text) for text in texts),
    )


def refuse_compile_work_over_bound(
    equations: Sequence[str], threshold: str | None, reset: str | None
) -> None:
    """Refuse an equation compile whose work is over the bound, before anything parses it.

    The characters are counted first, so an oversized request is refused without
    being parsed.

    Parameters
    ----------
    equations : Sequence[str]
        The request's equations.
    threshold : str or None
        The request's threshold.
    reset : str or None
        The request's reset.

    Raises
    ------
    CompileWorkRefused
        Over :data:`MAX_EXPRESSION_CHARACTERS` characters, or over
        :data:`MAX_FUNCTION_CALLS` function calls.
    """
    characters = sum(len(text) for text in _texts(equations, threshold, reset))
    if characters > MAX_EXPRESSION_CHARACTERS:
        raise CompileWorkRefused(
            f"the equations, threshold and reset hold {characters} characters; "
            f"one compile may hold at most {MAX_EXPRESSION_CHARACTERS}"
        )
    work = measure_compile_work(equations, threshold, reset)
    if work.function_calls > MAX_FUNCTION_CALLS:
        raise CompileWorkRefused(
            f"the equations, threshold and reset call {work.function_calls} functions; "
            f"one compile may call at most {MAX_FUNCTION_CALLS}"
        )


__all__ = [
    "MAX_EXPRESSION_CHARACTERS",
    "MAX_FUNCTION_CALLS",
    "CompileWork",
    "CompileWorkRefused",
    "measure_compile_work",
    "refuse_compile_work_over_bound",
]
