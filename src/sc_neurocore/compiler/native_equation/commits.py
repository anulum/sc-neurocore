# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native equation state and event overflow

"""Preserve the registered compiler's distinct integrated-state and event commits."""

from __future__ import annotations

from .arithmetic import EquationArithmetic


class EquationCommits:
    """Apply signed Q-word overflow at the maintained model's commit boundaries.

    Parameters
    ----------
    arithmetic : EquationArithmetic
        Destination graph and numerical configuration. Word width and overflow
        mode are captured when constructing this policy.

    Raises
    ------
    NotImplementedError
        Unsigned equation state is requested.
    ValueError
        The overflow mode is outside saturate, wrap and trap.

    Notes
    -----
    A final integrated state traps in simulation when requested. Reset and
    graded expressions wrap in trap mode, as in the existing registered core.
    Wide bounds use actual word constants, without an i64-literal restriction.
    """

    def __init__(self, arithmetic: EquationArithmetic) -> None:
        if not arithmetic.q.signed:
            raise NotImplementedError("unsigned equation state is not supported")
        if arithmetic.q.overflow not in {"saturate", "wrap", "trap"}:
            raise ValueError(f"Unknown overflow mode: {arithmetic.q.overflow!r}")
        self.arithmetic = arithmetic
        self.data_width = arithmetic.q.data_width
        self.overflow = arithmetic.q.overflow

    def state(self, raw: int, width: int) -> int:
        """Commit a final integrated word, including its requested simulation trap.

        Parameters
        ----------
        raw : int
            Signed native SSA word before narrowing.
        width : int
            Physical width of that word.

        Returns
        -------
        int
            Signed Q-word identifier after the declared overflow policy.
        """
        return self._commit(raw, width, trap=True)

    def event(self, raw: int, width: int) -> int:
        """Commit a reset or graded value; trap mode retains its existing wrapping.

        Parameters
        ----------
        raw : int
            Signed native SSA reset or graded-expression result.
        width : int
            Physical width of that word.

        Returns
        -------
        int
            Signed Q word, without introducing an event-only simulation trap.
        """
        return self._commit(raw, width, trap=False)

    def _commit(self, raw: int, width: int, *, trap: bool) -> int:
        """Append the declared bounds and narrowing without intermediate saturation."""
        if type(width) is not int or width < 1:
            raise ValueError("commit width must be a positive integer")
        a, b, dw = self.arithmetic, self.arithmetic.builder, self.data_width
        if width <= dw or self.overflow == "wrap" or self.overflow == "trap" and not trap:
            return b.word_resize(raw, f"i{dw}")
        ty = f"i{width}"
        minimum = a.constant(-(1 << (dw - 1)), width)
        maximum = a.constant((1 << (dw - 1)) - 1, width)
        lower = b.word_compare("ge", raw, minimum)
        upper = b.word_compare("le", raw, maximum)
        if self.overflow == "trap":
            valid = b.word_select(lower, upper, b.constant_i64(0, "bool"), "bool")
            b.runtime_check(valid, "overflow")
        else:
            raw = b.word_select(upper, raw, maximum, ty)
            raw = b.word_select(lower, raw, minimum, ty)
        return b.word_resize(raw, f"i{dw}")
