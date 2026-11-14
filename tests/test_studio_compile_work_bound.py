# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — One equation compile is bounded before the RTL emitter runs

"""An equation compile is refused in words, before the RTL emitter runs, when its work is over the bound.

Measured before this was written, on the tree that moved the RTL emit into the
check before admission: the emitter writes a lookup table for every function
call, about 11,000 characters of RTL each. One admissible equation, a balanced
sum of 1,600 ``tanh(v)`` calls in 16,005 characters, wrote 17.75 million
characters of RTL in the API process; by arithmetic, a 1 MiB body holds some 65
such equations. ``/api/ir/emit-sv-direct`` already emitted such an equation in the API
process. The largest legitimate source, over every bound catalogue schema,
Studio template and Studio preset, calls 25 functions in 2,333 characters; the
longest holds 3,204 characters.
"""

from collections.abc import Iterator
import json
from pathlib import Path
from typing import Any

import pytest

from sc_neurocore.studio.compile_work_bound import (
    MAX_EXPRESSION_CHARACTERS,
    MAX_FUNCTION_CALLS,
    CompileWorkRefused,
    measure_compile_work,
    refuse_compile_work_over_bound,
)
from tests.test_studio_ode_compile_refusals import VALID, post_compile, studio_client


def calls(count: int) -> str:
    """Return an ODE whose right side calls ``tanh`` ``count`` times.

    Parameters
    ----------
    count : int
        The number of calls.

    Returns
    -------
    str
        The equation.
    """
    return "dv/dt = " + " + ".join(["tanh(v)"] * count)


def calls_refusal(count: int) -> str:
    """Return the whole refusal of a request that calls ``count`` functions.

    Parameters
    ----------
    count : int
        The number of calls in the request.

    Returns
    -------
    str
        The refusal's text.
    """
    return (
        f"the equations, threshold and reset call {count} functions; "
        f"one compile may call at most {MAX_FUNCTION_CALLS}"
    )


def test_the_bound_is_the_one_stated() -> None:
    """128 function calls and 16,384 characters, as stated for the routes."""
    assert (MAX_FUNCTION_CALLS, MAX_EXPRESSION_CHARACTERS) == (128, 16_384)


def test_a_request_at_the_bounds_is_admitted() -> None:
    """Exactly the bound on calls, and exactly the bound on characters, are admitted."""
    refuse_compile_work_over_bound([calls(MAX_FUNCTION_CALLS)], None, None)
    padded = "dv/dt = -v" + " " * (MAX_EXPRESSION_CHARACTERS - len("dv/dt = -v"))
    assert measure_compile_work([padded], None, None).characters == MAX_EXPRESSION_CHARACTERS
    refuse_compile_work_over_bound([padded], None, None)


def test_one_call_over_the_bound_is_refused_in_words() -> None:
    """The refusal names the request's calls and the bound."""
    with pytest.raises(CompileWorkRefused) as refused:
        refuse_compile_work_over_bound([calls(MAX_FUNCTION_CALLS + 1)], None, None)
    assert str(refused.value) == calls_refusal(MAX_FUNCTION_CALLS + 1)


def test_one_character_over_the_bound_is_refused_in_words() -> None:
    """The characters are counted over the equations, the threshold and the reset together."""
    equation = "dv/dt = -v"
    threshold = "v > 1" + " " * (MAX_EXPRESSION_CHARACTERS - len(equation) - len("v > 1") - 4)
    assert measure_compile_work([equation], threshold, "v= 0").characters == (
        MAX_EXPRESSION_CHARACTERS
    )
    refuse_compile_work_over_bound([equation], threshold, "v= 0")
    with pytest.raises(CompileWorkRefused) as refused:
        refuse_compile_work_over_bound([equation], threshold, "v = 0")
    assert str(refused.value) == (
        f"the equations, threshold and reset hold {MAX_EXPRESSION_CHARACTERS + 1} characters; "
        f"one compile may hold at most {MAX_EXPRESSION_CHARACTERS}"
    )


@pytest.mark.parametrize(
    ("equations", "threshold", "reset"),
    [
        ([calls(43), calls(43).replace("dv/dt", "dw/dt")], " or ".join(["exp(v) > 1"] * 43), None),
        ([calls(43)], " or ".join(["exp(v) > 1"] * 43), "; ".join(["v = tanh(v)"] * 43)),
        ([f"{calls(64)} ; {calls(65).replace('dv/dt', 'dw/dt')}"], None, None),
    ],
    ids=["two-equations-and-threshold", "equation-threshold-reset", "semicolon-joined"],
)
def test_calls_are_counted_over_every_expression_the_compile_parses(
    equations: list[str], threshold: str | None, reset: str | None
) -> None:
    """Calls in each equation, the threshold and each reset rule add up to one count.

    Parameters
    ----------
    equations : list[str]
        The request's equations.
    threshold : str or None
        The request's threshold.
    reset : str or None
        The request's reset.
    """
    assert measure_compile_work(equations, threshold, reset).function_calls == 129
    with pytest.raises(CompileWorkRefused) as refused:
        refuse_compile_work_over_bound(equations, threshold, reset)
    assert str(refused.value) == calls_refusal(129)


def test_text_the_count_cannot_parse_is_bounded_by_its_parentheses() -> None:
    """A part that does not parse counts each of its opening parentheses as a call."""
    unreadable = "dv/dt = " + " + ".join(["tanh(v)"] * 129) + " +"
    assert measure_compile_work([unreadable], None, None).function_calls == 129
    with pytest.raises(CompileWorkRefused):
        refuse_compile_work_over_bound([unreadable], None, None)


def test_grouping_parentheses_in_joined_parts_are_not_calls() -> None:
    """Each ``;``-joined part is read on its own, so its grouping parentheses are not counted."""
    grouped = "(" * 65 + "v" + ")" * 65
    joined = f"dv/dt = -{grouped} ; dw/dt = -{grouped.replace('v', 'w')}"
    assert measure_compile_work([joined], None, None).function_calls == 0
    refuse_compile_work_over_bound([joined], None, None)


def test_a_call_either_reading_holds_is_counted() -> None:
    """A call one reading hides in a comment, and the other reads, is counted."""
    assert measure_compile_work([], "v > 1 # = tanh(v)", None).function_calls == 1


def _catalogue_sources() -> Iterator[tuple[str, list[str], str | None, str | None]]:
    """Yield the equations, threshold and reset of every bound catalogue schema.

    Yields
    ------
    tuple[str, list[str], str or None, str or None]
        The model's name, its equations, its threshold condition and its reset rules.
    """
    from sc_neurocore.neurons.models import _CLASS_TO_MODULE
    from sc_neurocore.studio.candidate_catalogue import (
        CatalogueCandidateUnavailable,
        candidate_from_catalogue,
    )

    for name in sorted(_CLASS_TO_MODULE):
        try:
            schema = json.loads(candidate_from_catalogue(name)["candidate_text"])["model"]
        except CatalogueCandidateUnavailable:
            continue
        dynamics: dict[str, Any] = schema.get("dynamics") or {}
        reset: dict[str, Any] = schema.get("reset") or {}
        yield (
            name,
            [f"d{var}/dt = {rhs}" for var, rhs in dynamics.items()],
            (schema.get("threshold") or {}).get("condition"),
            "; ".join(f"{var} = {rule}" for var, rule in reset.items()) or None,
        )


def _studio_sources() -> Iterator[tuple[str, list[str], str | None, str | None]]:
    """Yield the equations, threshold and reset of every Studio template and preset.

    Yields
    ------
    tuple[str, list[str], str or None, str or None]
        The source's name, its equations, its threshold and its reset.
    """
    from sc_neurocore.studio.presets import PRESETS
    from sc_neurocore.studio.templates import TEMPLATES

    named = [(f"template:{key}", value) for key, value in TEMPLATES.items()]
    named += [(f"preset:{index}", value) for index, value in enumerate(PRESETS)]
    for name, source in named:
        yield name, list(source.get("equations", [])), source.get("threshold"), source.get("reset")


def test_every_legitimate_source_is_within_the_bound() -> None:
    """No bound catalogue schema, Studio template or Studio preset is refused."""
    sources = [*_catalogue_sources(), *_studio_sources()]
    assert len(sources) >= 90
    over = {}
    for name, equations, threshold, reset in sources:
        work = measure_compile_work(equations, threshold, reset)
        if work.function_calls > MAX_FUNCTION_CALLS or work.characters > MAX_EXPRESSION_CHARACTERS:
            over[name] = work
    assert over == {}


def test_the_compile_task_refuses_over_the_bound_before_it_emits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared compile refuses before the emitter is called, so the job is bounded too.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Records any call of the emitter.
    """
    import sc_neurocore.studio.platform.compile_process as compile_process

    emitted: list[object] = []
    monkeypatch.setattr(compile_process, "equation_to_fpga", lambda *a, **k: emitted.append(a))
    payload = {**VALID, "equations": [calls(MAX_FUNCTION_CALLS + 1)]}
    with pytest.raises(CompileWorkRefused):
        compile_process.compile_equation_payload(payload)
    assert emitted == []


def test_the_direct_emit_refuses_over_the_bound_before_it_emits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``/api/ir/emit-sv-direct``'s function refuses before the emitter is called.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Records any call of the emitter.
    """
    import sc_neurocore.compiler.equation_compiler as equation_compiler
    from sc_neurocore.studio.compiler import emit_sv_from_equation

    emitted: list[object] = []
    monkeypatch.setattr(equation_compiler, "equation_to_fpga", lambda *a, **k: emitted.append(a))
    with pytest.raises(CompileWorkRefused) as refused:
        emit_sv_from_equation(equations=[calls(MAX_FUNCTION_CALLS + 1)])
    assert str(refused.value) == calls_refusal(MAX_FUNCTION_CALLS + 1)
    assert emitted == []


def test_the_compile_route_answers_422_with_the_bound_and_runs_no_job(tmp_path: Path) -> None:
    """``/api/compile`` tells the caller the bound, and admits no job.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test-owned jobs and audit location.
    """
    with studio_client(tmp_path) as client:
        before = client.get("/api/studio/jobs").json()
        response = post_compile(client, {**VALID, "equations": [calls(MAX_FUNCTION_CALLS + 1)]})
        assert response.status_code == 422, response.text
        assert response.json() == {"detail": calls_refusal(MAX_FUNCTION_CALLS + 1)}
        assert client.get("/api/studio/jobs").json() == before


def test_the_direct_emit_route_answers_422(tmp_path: Path) -> None:
    """``/api/ir/emit-sv-direct`` refuses as it refuses every input it cannot emit.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test-owned jobs and audit location.
    """
    with studio_client(tmp_path) as client:
        response = client.post(
            "/api/ir/emit-sv-direct",
            json={**VALID, "equations": [calls(MAX_FUNCTION_CALLS + 1)]},
            headers={"Origin": "http://127.0.0.1:5173"},
        )
        assert response.status_code == 422, response.text
        assert response.json() == {"detail": "Invalid input"}
