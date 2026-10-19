# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — rand()/randn() in equation-defined neurons

"""Tests of :mod:`sc_neurocore.neurons.equation_random` and its use by :class:`EquationNeuron`."""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.equation_builder import EquationNeuron
from sc_neurocore.neurons.equation_random import RandomDraws, number_random_sites
from sc_neurocore.neurons.equation_refusals import EquationRefusal


class TestSiteNumbering:
    """Bare calls get their own sites; explicit sites are kept; other text is untouched."""

    def test_numbering_order_and_explicit_sites(self) -> None:
        """Smallest free numbers in expression and source order, around an explicit site."""
        assert number_random_sites(
            ["-v + 0.1*randn()", None, "v > rand() + rand(0)", "rand()*randn()"]
        ) == [
            "-v + 0.1*randn(1)",
            None,
            "v > rand(2) + rand(0)",
            "rand(3)*randn(4)",
        ]

    def test_poisson_sites_follow_the_mean(self) -> None:
        """``poisson(lam)`` gains its site as a second argument; an explicit one is kept."""
        assert number_random_sites(
            ["poisson(3.0) + rand()", "poisson(2.5, 7)", "poisson(1.0,)"]
        ) == [
            "poisson(3.0, 0) + rand(1)",
            "poisson(2.5, 7)",
            "poisson(1.0, 2)",
        ]

    def test_expression_without_draws_is_unchanged(self) -> None:
        """Formatting of an expression without draws is preserved exactly."""
        assert number_random_sites(["-v  +   I*2"]) == ["-v  +   I*2"]

    def test_unparseable_text_is_left_to_the_validator(self) -> None:
        """Invalid syntax passes through unchanged so the equation validator reports it."""
        assert number_random_sites(["v +* rand()"]) == ["v +* rand()"]

    @pytest.mark.parametrize(
        "expressions",
        [
            ["rand(1) + rand(1)"],
            ["rand(-1)"],
            ["rand(1.5)"],
            ["rand(1, 2)"],
            ["poisson()"],
            ["poisson(1.0, 2, 3)"],
            ["rand(site=1)"],
        ],
    )
    def test_invalid_explicit_sites_are_refused(self, expressions: list[str]) -> None:
        """A repeated, negative or non-integer site, a wrong count or a keyword is refused."""
        with pytest.raises(ValueError):
            number_random_sites(expressions)


class TestRandomDraws:
    """Seed and stream ranges, and draws as pure functions of the counter."""

    @pytest.mark.parametrize(("seed", "stream"), [(-1, 0), (1 << 64, 0), (0, 1 << 32), (True, 0)])
    def test_ranges_are_checked(self, seed: int, stream: int) -> None:
        """Seeds outside 64 bits and streams outside 32 bits are refused."""
        with pytest.raises(ValueError):
            RandomDraws(seed, stream)

    def test_draw_depends_on_site_step_stream_and_seed(self) -> None:
        """Changing any counter part or the seed changes the draw; repeating does not."""
        draws = RandomDraws(5, 0)
        first = draws.rand(0)
        assert draws.rand(0) == first
        assert (
            draws.rand(1) != first
            and RandomDraws(6, 0).rand(0) != first
            and RandomDraws(5, 1).rand(0) != first
        )
        draws.step = 1
        assert draws.rand(0) != first


class TestNeuron:
    """EquationNeuron numbers its draws, advances the counter per step and resets it."""

    def test_neuron_rewrites_its_expressions_and_is_reproducible(self) -> None:
        """Two neurons with the same seed produce the same trace; reset replays it."""

        def make() -> EquationNeuron:
            return EquationNeuron(
                equations={"v": "-v + 0.5*randn()"},
                state={"v": 0.0},
                dt=0.1,
                threshold="v > 0.2",
                reset={"v": "rand() - 0.5"},
                rng_seed=99,
                random_stream=3,
            )

        a, b = make(), make()
        assert a.equations == {"v": "-v + 0.5*randn(0)"} and a.reset_rules == {"v": "rand(1) - 0.5"}
        trace_a = [(a.step(0.0), a.state["v"]) for _ in range(50)]
        assert trace_a == [(b.step(0.0), b.state["v"]) for _ in range(50)]
        a.reset()
        assert [(a.step(0.0), a.state["v"]) for _ in range(50)] == trace_a
        assert (a.random_seed, a.random_stream) == (99, 3)

    def test_stream_separates_instances(self) -> None:
        """Two instances with different streams draw different numbers from the same seed."""
        first = EquationNeuron(equations={"v": "rand()"}, method="map", dt=1.0, random_stream=0)
        second = EquationNeuron(equations={"v": "rand()"}, method="map", dt=1.0, random_stream=1)
        first.step(0.0)
        second.step(0.0)
        assert first.state["v"] != second.state["v"]

    def test_invalid_seed_is_an_equation_refusal(self) -> None:
        """A seed outside 64 bits is refused when the neuron is built."""
        with pytest.raises(EquationRefusal):
            EquationNeuron(equations={"v": "rand()"}, method="map", dt=1.0, rng_seed=-1)
