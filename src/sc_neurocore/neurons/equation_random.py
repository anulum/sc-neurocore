# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — rand() and randn() in equations: call-site numbering and the neuron's draws

"""Counter-based random draws for equation-defined neurons.

Every ``rand()``, ``randn()``, ``poisson(lam)`` or ``poisson_grid(lam)`` call in a
neuron's expressions draws one
Philox4x32-10 block (:mod:`sc_neurocore.compiler.random_units`) at counter
``(site, step, stream)`` with the neuron's seed. The call site is numbered when
the neuron is built, :func:`number_random_sites` rewriting ``rand()`` to
``rand(<site>)`` and ``poisson(lam)`` to ``poisson(lam, <site>)`` (likewise
``poisson_grid``), so the Python
model and the equation compiler read the same number from the same text. An
explicit integer literal is kept as written.
Repeating a step's evaluation (a Runge–Kutta stage, a reset expression) draws
the same number; the step counter advances once per :meth:`EquationNeuron.step`.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from typing import cast

RANDOM_CALLS = frozenset({"rand", "randn", "poisson", "poisson_grid"})

XI_SITE = (1 << 32) - 1
"""Call site of the diffusion-noise symbol ``xi``; sub-step ``k`` of a step reads ``XI_SITE - k``."""

RESERVED_SITES = 1 << 31
"""Sites from here up are reserved (``xi``, the Philox threshold trial); explicit sites lie below."""

STOCHASTIC_SITE = RESERVED_SITES
"""Call site of the Philox stochastic-threshold trial (``threshold_rng="philox"``)."""

# Arguments before the call site: poisson's mean.
_LEADING = {"rand": 0, "randn": 0, "poisson": 1, "poisson_grid": 1}


def _calls(expression: str) -> list[tuple[str, ast.Call]]:
    """Return the ``rand``/``randn`` calls of an expression with their names, in source order.

    Text that does not parse has no calls here; the equation validator reports it.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return []
    calls = [
        (node.func.id, node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in RANDOM_CALLS
    ]
    return sorted(calls, key=lambda pair: (pair[1].lineno, pair[1].col_offset))


def draws_in(expressions: Sequence[str | None]) -> bool:
    """Return whether any of the expressions calls ``rand``, ``randn`` or ``poisson``.

    Parameters
    ----------
    expressions : Sequence[str or None]
        Expressions to inspect; ``None`` entries are skipped.

    Returns
    -------
    bool
        ``True`` when at least one random call is present. The test reads the
        parsed calls, so a state or parameter whose name merely contains
        ``rand`` or ``rng_step`` does not count.

    """
    return any(_calls(expression) for expression in expressions if expression is not None)


def number_random_sites(expressions: Sequence[str | None]) -> list[str | None]:
    """Give every bare ``rand()``/``randn()`` call its own call-site number.

    Parameters
    ----------
    expressions : Sequence[str or None]
        The neuron's expressions in a fixed order; ``None`` entries are kept.

    Returns
    -------
    list[str or None]
        The expressions with ``rand()`` rewritten to ``rand(<site>)`` and
        ``poisson(lam)`` to ``poisson(lam, <site>)``. Sites are the
        smallest non-negative integers not already written explicitly, in expression
        order and, within an expression, in source order. Expressions without such
        calls are returned unchanged, character for character.

    Raises
    ------
    ValueError
        If a call has the wrong number of arguments or a keyword, an explicit
        site is not a non-negative integer literal, or one site is written twice.

    """
    used: set[int] = set()
    for expression in expressions:
        if expression is None:
            continue
        for name, call in _calls(expression):
            leading = _LEADING[name]
            if call.keywords or len(call.args) not in (leading, leading + 1):
                raise ValueError(
                    f"{name}() takes {leading} argument(s) and an optional call-site literal"
                )
            if len(call.args) == leading:
                continue
            site = call.args[leading]
            if (
                not isinstance(site, ast.Constant)
                or isinstance(site.value, bool)
                or not isinstance(site.value, int)
                or site.value < 0
            ):
                raise ValueError(f"{name}() takes an optional call-site integer literal")
            if site.value >= RESERVED_SITES:
                raise ValueError(
                    f"random call site {site.value} is reserved for the diffusion noise xi; "
                    f"explicit sites must be below {RESERVED_SITES}"
                )
            if site.value in used:
                raise ValueError(f"random call site {site.value} is used twice")
            used.add(site.value)
    next_site = 0
    result: list[str | None] = []
    for expression in expressions:
        if expression is None:
            result.append(None)
            continue
        bare = [call for name, call in _calls(expression) if len(call.args) == _LEADING[name]]
        if not bare:
            result.append(expression)
            continue
        lines = expression.split("\n")
        offsets = [0]
        for line in lines[:-1]:
            offsets.append(offsets[-1] + len(line.encode()) + 1)
        text = expression.encode()
        inserts: list[tuple[int, bytes]] = []
        for call in bare:
            while next_site in used:
                next_site += 1
            # ast.parse sets the end position of every node it returns.
            end_line = cast(int, call.end_lineno)
            end_column = cast(int, call.end_col_offset)
            position = offsets[end_line - 1] + end_column - 1
            inside = text[:position].rstrip()
            separator = b"" if not call.args else b" " if inside.endswith(b",") else b", "
            inserts.append((position, separator + str(next_site).encode()))
            used.add(next_site)
        for position, insertion in sorted(inserts, reverse=True):
            text = text[:position] + insertion + text[position:]
        result.append(text.decode())
    return result


class RandomDraws:
    """The draws of one neuron: its seed, its stream and its step counter.

    Parameters
    ----------
    seed : int
        Philox key, ``[0, 2**64)``.
    stream : int
        Instance number, ``[0, 2**32)``.

    """

    def __init__(self, seed: int, stream: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 1 << 64:
            raise ValueError(f"random seed must be an integer in [0, 2**64), got {seed!r}")
        if isinstance(stream, bool) or not isinstance(stream, int) or not 0 <= stream < 1 << 32:
            raise ValueError(f"random stream must be an integer in [0, 2**32), got {stream!r}")
        self.seed = seed
        self.stream = stream
        self.step = 0

    def rand(self, site: int) -> float:
        """Return the uniform draw of a call site at the current step, in ``[0, 1)``.

        Parameters
        ----------
        site : int
            Call-site number.

        Returns
        -------
        float
            Word 0 of the block as a fraction of ``2**32``.

        """
        # Imported here: the compiler package imports this package at load time.
        from sc_neurocore.compiler import random_units as ru

        return ru.uniform_value(ru.draw(site, self.step, self.stream, self.seed)[0])

    def randn(self, site: int) -> float:
        """Return the standard-normal draw of a call site at the current step.

        Parameters
        ----------
        site : int
            Call-site number.

        Returns
        -------
        float
            Box–Muller of words 0 and 1 of the block.

        """
        from sc_neurocore.compiler import random_units as ru

        words = ru.draw(site, self.step, self.stream, self.seed)
        return ru.normal_value(words[0], words[1])

    def poisson(self, lam: float, site: int) -> float:
        """Return the Poisson draw of a call site at the current step.

        Parameters
        ----------
        lam : float
            The mean, ``[0, MAX_POISSON_LAMBDA]`` of
            :mod:`sc_neurocore.compiler.random_units`.
        site : int
            Call-site number.

        Returns
        -------
        float
            The count from word 0 by exact inversion.

        """
        from sc_neurocore.compiler import random_units as ru

        return float(ru.poisson_count(ru.draw(site, self.step, self.stream, self.seed)[0], lam))

    def poisson_grid(self, lam: float, site: int) -> float:
        """Return the Poisson draw of a call site for a mean floored to the 1/16 grid.

        Parameters
        ----------
        lam : float
            The mean; floored to a multiple of 1/16 and clamped to ``[0, 16]``.
        site : int
            Call-site number.

        Returns
        -------
        float
            The exact inversion of word 0 for the grid mean, as the compiled unit
            computes it.

        """
        from sc_neurocore.compiler import random_units as ru

        word = ru.draw(site, self.step, self.stream, self.seed)[0]
        return float(ru.poisson_grid_count(word, ru.poisson_grid_index(lam)))


__all__ = [
    "RANDOM_CALLS",
    "RESERVED_SITES",
    "STOCHASTIC_SITE",
    "XI_SITE",
    "PhiloxThreshold",
    "RandomDraws",
    "draws_in",
    "number_random_sites",
]


class PhiloxThreshold:
    """The Philox stochastic-threshold trial: fire when ``rand(STOCHASTIC_SITE) < p``.

    The counter-based alternative to the 16-bit LFSR (``threshold_rng="philox"``):
    the trial reads word 0 of the neuron's own draw at the reserved call site and
    the current step, so every neuron of a population has its own stream and the
    trial resolves ``p`` to ``2**-32`` instead of ``1/65535``. The compiler lowers
    the same comparison as a level threshold, which a Q-format with ``f`` fraction
    bits evaluates as ``floor(word0 / 2**(32 - f)) < P``: the same decision
    whenever ``p`` is exactly the code ``P / 2**f``.

    Parameters
    ----------
    draws : RandomDraws
        The neuron's draws (seed, stream and step counter).
    """

    def __init__(self, draws: RandomDraws) -> None:
        self._draws = draws

    @property
    def initial_seed(self) -> int:
        """Return the Philox key of the trials."""
        return self._draws.seed

    @property
    def state(self) -> int:
        """Return the step counter the next trial reads."""
        return self._draws.step

    def trial(self, probability: float) -> bool:
        """Return whether this step's trial fires with ``probability``."""
        return self._draws.rand(STOCHASTIC_SITE) < probability

    def restore(self, state: int) -> None:
        """Return the step counter to ``state`` (a refused step draws again)."""
        self._draws.step = state

    def reset(self) -> None:
        """Leave the counter to the neuron's reset, which returns it to step 0."""
