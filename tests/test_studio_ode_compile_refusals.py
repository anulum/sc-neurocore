# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — A refused equation compile says why, before any job runs

"""An equation compile the parser refuses is answered 422 with its reason, not a failed job.

Measured before this was written: with the reset garbled to
``v = -65 = (-v + 1) / 10``, the ODE editor's 'Generate RTL' (``POST
/api/compile``) was answered HTTP 500 ``studio_job_failed``, with the public
error "Studio job failed.", while the job ledger kept the parser's reason. A
garbled threshold and an equation that is not an ODE were answered the same
way. Each is refused by the equation parse the worker runs before the RTL
emitter, with text written for the caller. The RTL emitter's own refusals (a name
the RTL has no signal for, an unknown function, a reset of a name that is not a
state) were answered the same way; the check now runs the worker's whole compile.
"""

from pathlib import Path
from typing import Any

from fastapi import HTTPException
import httpx2
import pytest
from starlette.testclient import TestClient

from sc_neurocore.refusals import AuthoredRefusal
from sc_neurocore.studio.api.compiler import refuse_equations_before_admission
from sc_neurocore.studio.api.schemas import CompileRequest
from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import StudioRuntimeSettings
from sc_neurocore.studio.platform.compile_process import compile_equation_payload

#: The ODE editor's request with every field valid; each refusal changes one field.
VALID: dict[str, Any] = {
    "equations": ["dv/dt = (-v + 1) / 10"],
    "threshold": "v > 0.9",
    "reset": "v = 0",
    "params": {},
    "init": {"v": 0.0},
}

#: One garbled field each, with the whole reason the caller must be given.
REFUSALS: list[tuple[dict[str, Any], str]] = [
    ({"reset": "v = -65 = (-v + 1) / 10"}, "Invalid equation syntax: '-65 = (-v + 1) / 10'"),
    ({"threshold": "v >"}, "Invalid equation syntax: 'v >'"),
    ({"equations": ["v = 1"]}, "Cannot parse equation: 'v = 1'. Expected 'd<var>/dt = <expr>'"),
    (
        {"equations": ["dv/dt = -v / tau"]},
        "the equation reads 'tau', which is not a state, a parameter or a constant of "
        "this neuron; the RTL has no signal for it",
    ),
    (
        {"equations": ["dv/dt = frobnicate(v)"]},
        "Unsupported function 'frobnicate' in Verilog compilation.",
    ),
    ({"reset": "w = 0"}, "the reset assigns 'w', which is not a state of this neuron"),
]


def studio_client(tmp_path: Path) -> TestClient:
    """Build a Studio client whose jobs and audit log live under the test's directory.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test-owned jobs and audit location.

    Returns
    -------
    starlette.testclient.TestClient
        A client for an app with test-owned runtime settings.
    """
    settings = StudioRuntimeSettings(
        job_root_path=str(tmp_path / "jobs"),
        audit_log_path=str(tmp_path / "audit.jsonl"),
    )
    return TestClient(create_app(settings), base_url="http://127.0.0.1")


def post_compile(client: TestClient, request: dict[str, Any]) -> httpx2.Response:
    """Send an equation compile the way the ODE editor's 'Generate RTL' does.

    Parameters
    ----------
    client : starlette.testclient.TestClient
        The Studio client.
    request : dict[str, Any]
        The compile request body.

    Returns
    -------
    httpx2.Response
        The HTTP response.
    """
    return client.post("/api/compile", json=request, headers={"Origin": "http://127.0.0.1:5173"})


@pytest.mark.parametrize(("change", "reason"), REFUSALS)
def test_a_refused_equation_compile_answers_422_with_its_reason_and_runs_no_job(
    tmp_path: Path, change: dict[str, Any], reason: str
) -> None:
    """The caller is told what the parser refused, and no job is admitted for it.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test-owned jobs and audit location.
    change : dict[str, Any]
        The garbled field.
    reason : str
        The whole reason the caller must be given.
    """
    with studio_client(tmp_path) as client:
        before = client.get("/api/studio/jobs").json()
        response = post_compile(client, {**VALID, **change})
        assert response.status_code == 422, response.text
        assert response.json() == {"detail": reason}
        assert client.get("/api/studio/jobs").json() == before


@pytest.mark.parametrize(("change", "reason"), REFUSALS)
def test_the_check_refuses_what_the_worker_compile_refuses(
    change: dict[str, Any], reason: str
) -> None:
    """The check and the worker's compile refuse the same payload with the same text.

    The worker's refusal is an authored one, the class the check answers 422.

    Parameters
    ----------
    change : dict[str, Any]
        The garbled field.
    reason : str
        The whole refusal message.
    """
    request = CompileRequest(**{**VALID, **change})
    with pytest.raises(AuthoredRefusal) as compiled:
        compile_equation_payload(request.model_dump())
    with pytest.raises(HTTPException) as checked:
        refuse_equations_before_admission(request)
    assert checked.value.status_code == 422
    assert checked.value.detail == str(compiled.value) == reason


@pytest.mark.parametrize(
    "request_body",
    [
        VALID,
        {**VALID, "equations": [" dv/dt = -w / tau ;; dw/dt = v ; "], "params": {"tau": 10.0}},
        {**VALID, "module_name": "sc_named_module"},
    ],
    ids=["one-ode", "semicolon-joined", "module-name"],
)
def test_the_check_runs_the_worker_compile_on_the_payload_the_job_receives(
    monkeypatch: pytest.MonkeyPatch, request_body: dict[str, Any]
) -> None:
    """The check compiles exactly what the job is given, so it never refuses what the job accepts.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Records what the check compiles.
    request_body : dict[str, Any]
        A request the worker compiles.
    """
    import sc_neurocore.studio.api.compiler as compiler_routes

    seen: list[object] = []

    def recording_compile(payload: Any) -> Any:
        seen.append(payload)
        return compile_equation_payload(payload)

    monkeypatch.setattr(compiler_routes, "compile_equation_payload", recording_compile)
    request = CompileRequest(**request_body)
    refuse_equations_before_admission(request)
    assert seen == [request.model_dump()]


def test_the_compile_task_names_the_module_it_is_given() -> None:
    """The shared compile emits the module under the payload's name, as the job returns it."""
    request = CompileRequest(**{**VALID, "module_name": "sc_named_module"})
    compiled, verilog = compile_equation_payload(request.model_dump())
    assert compiled.module_name == "sc_named_module"
    assert "module sc_named_module" in verilog


@pytest.mark.parametrize(
    "fault",
    [
        ValueError("a plain parse fault"),
        KeyError("w"),
        TypeError("unsupported operand type(s)"),
        ZeroDivisionError("float division by zero"),
        FileNotFoundError("units table"),
    ],
)
def test_a_fault_that_is_not_an_authored_refusal_is_left_to_the_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: Exception
) -> None:
    """A fault that is not written for the caller does not answer for the compiler.

    The check lets the job run, and the job records the fault as before. It
    must not reach the request boundary, which would answer a ``KeyError`` or
    a ``TypeError`` 422 "Invalid input", as if the caller had sent it.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test-owned jobs and audit location.
    monkeypatch : pytest.MonkeyPatch
        Makes the check's compile meet such a fault; the worker compiles on its own.
    fault : Exception
        What the check's compile raises.
    """
    import sc_neurocore.studio.api.compiler as compiler_routes

    def faulty_compile(*_args: object, **_kwargs: object) -> None:
        raise fault

    monkeypatch.setattr(compiler_routes, "compile_equation_payload", faulty_compile)
    with studio_client(tmp_path) as client:
        response = post_compile(client, VALID)
        assert response.status_code == 200, response.text


@pytest.mark.parametrize(
    "request_body",
    [
        VALID,
        {**VALID, "equations": ["dv/dt = -w; dw/dt = v"], "init": {"v": 0.0, "w": 0.0}},
    ],
    ids=["one-ode", "semicolon-joined"],
)
def test_an_equation_compile_the_parser_accepts_still_runs(
    tmp_path: Path, request_body: dict[str, Any]
) -> None:
    """A request the worker compiles is not stopped by the check.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test-owned jobs and audit location.
    request_body : dict[str, Any]
        A valid request; the second joins two ODEs with ';' in one string.
    """
    with studio_client(tmp_path) as client:
        response = post_compile(client, request_body)
        assert response.status_code == 200, response.text
        assert "module sc_neuron" in response.json()["verilog"]
