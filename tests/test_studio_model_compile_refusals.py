# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — A refused model compile says why, before any job runs

"""A compile request the compiler refuses is answered 422 with its reason, not a failed job.

Measured before this was written: every compile the compiler refused was
answered HTTP 500 ``studio_job_failed``, with the public error "Studio job
failed.", while the job ledger kept the reason. Sent as the Studio sends it,
at each model's own step, 10 of the 64 catalogue models that offer 'Generate
RTL' were refused: nine by parameter names their schema does not know, one by
a default Q-format that cannot hold it. A map is refused too when it is
compiled at a step other than its own. One model of each kind is covered here.
"""

from pathlib import Path
import re

import pytest
from starlette.testclient import TestClient

from sc_neurocore.neurons.model_profile import ProfileAdmissionError
from sc_neurocore.refusals import AuthoredRefusal
from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.model_catalogue import get_model_detail
from sc_neurocore.studio.model_compile_configuration import (
    ModelCompileRefused,
    resolve_model_compile_configuration,
)
from sc_neurocore.studio.platform import StudioRuntimeSettings


def studio_request(
    model_name: str, *, dt: float | None = None, extra: dict[str, float] | None = None
) -> dict[str, object]:
    """Build a compile request the way the Studio's 'Generate RTL' does.

    Parameters
    ----------
    model_name : str
        A catalogue model with an RTL path.
    dt : float or None
        The step to send. None sends the model's own step, which the Studio
        takes on when the model is selected.
    extra : dict of str to float, optional
        Parameters added to the catalogue's, such as a name no side knows.

    Returns
    -------
    dict[str, object]
        The catalogue defaults, the default integrator and Q-format, and the step.
    """
    detail = get_model_detail(model_name)
    assert detail is not None
    configuration = detail["compile_configuration"]
    params = {parameter["name"]: parameter["default"] for parameter in detail["params"]}
    params.update(extra or {})
    return {
        "model_name": model_name,
        "dt": detail["dt"] if dt is None else dt,
        "integrator": configuration["default_integrator"],
        "q_format": configuration["default_q_format"],
        "params": params,
    }


#: One model per kind of refusal: the reason's opening, the step sent (None: the model's own) and any
#: parameter added. The map is sent 0.1, a step other than its declared 1.0, which the reader can set in
#: the Studio; AdEx is sent a parameter name neither its catalogue nor its schema knows.
REFUSALS = [
    ("AdExNeuron", "Unknown schema parameter override(s): bogus", None, {"bogus": 1.0}),
    ("RulkovMapNeuron", "recurrence 'Rulkov Map' has no continuous timebase", 0.1, None),
    ("SCScaledResetAdaptiveIFNeuron", "Q8.8 cannot hold this neuron", None, None),
]


@pytest.mark.parametrize("route", ["/api/models/compile", "/api/models/cosim"])
@pytest.mark.parametrize(("model_name", "reason", "dt", "extra"), REFUSALS)
def test_a_refused_compile_answers_422_with_its_reason_and_runs_no_job(
    tmp_path: Path,
    route: str,
    model_name: str,
    reason: str,
    dt: float | None,
    extra: dict[str, float] | None,
) -> None:
    """The caller is told what the compiler refused, and no job is admitted for it.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test-owned jobs and audit location.
    route : str
        The compile or the co-simulation request surface, which resolve alike.
    model_name : str
        A model whose Studio request the compiler refuses.
    reason : str
        The opening of the reason the caller must be given.
    dt : float or None
        The step sent; None sends the model's own.
    extra : dict of str to float or None
        Parameters added to the catalogue's.
    """
    settings = StudioRuntimeSettings(
        job_root_path=str(tmp_path / "jobs"),
        audit_log_path=str(tmp_path / "audit.jsonl"),
    )
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        before = client.get("/api/studio/jobs").json()
        response = client.post(
            route,
            json=studio_request(model_name, dt=dt, extra=extra),
            headers={"Origin": "http://127.0.0.1:5173"},
        )
        assert response.status_code == 422, response.text
        detail = response.json()["detail"]
        assert isinstance(detail, str)
        assert detail.startswith(reason)
        assert client.get("/api/studio/jobs").json() == before


@pytest.mark.parametrize(("model_name", "reason", "dt", "extra"), REFUSALS)
def test_each_refusal_is_an_authored_compile_refusal(
    model_name: str, reason: str, dt: float | None, extra: dict[str, float] | None
) -> None:
    """The resolver marks each refusal as text written for the caller.

    Parameters
    ----------
    model_name : str
        A model whose Studio request the compiler refuses.
    reason : str
        The opening of the refusal's message.
    dt : float or None
        The step sent; None sends the model's own.
    extra : dict of str to float or None
        Parameters added to the catalogue's.
    """
    with pytest.raises(ValueError, match=re.escape(reason)) as refused:
        resolve_model_compile_configuration(
            studio_request(model_name, dt=dt, extra=extra), detail_getter=get_model_detail
        )
    # The request boundary answers 422 for exactly this class. A map's missing
    # timebase is refused by the profile admission; the others by the resolver.
    assert isinstance(refused.value, AuthoredRefusal)
    expected = ProfileAdmissionError if model_name == "RulkovMapNeuron" else ModelCompileRefused
    assert type(refused.value) is expected


@pytest.mark.parametrize(
    "fault",
    [
        ValueError("Studio model compile configuration for 'X' has invalid integration."),
        KeyError("integration"),
        TypeError("unsupported operand type(s)"),
        FileNotFoundError("schema file"),
        ZeroDivisionError("float division by zero"),
    ],
)
def test_a_server_side_configuration_fault_is_left_to_the_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: Exception
) -> None:
    """A fault that is not an authored refusal does not answer for the compiler.

    It is a fault in the server's own compile configuration or code, not the
    caller's input, so the check lets the job run, and the job records it as
    before. It must not reach the request boundary, which would answer a
    ``KeyError`` or ``TypeError`` 422 "Invalid input", as if the caller had
    sent it.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test-owned jobs and audit location.
    monkeypatch : pytest.MonkeyPatch
        Makes the check's resolver meet such a fault; the worker resolves on its own.
    fault : Exception
        What the check's resolver raises.
    """
    import sc_neurocore.studio.api.compiler as compiler_routes

    def faulty_resolver(*_args: object, **_kwargs: object) -> None:
        raise fault

    monkeypatch.setattr(compiler_routes, "resolve_model_compile_configuration", faulty_resolver)
    settings = StudioRuntimeSettings(
        job_root_path=str(tmp_path / "jobs"),
        audit_log_path=str(tmp_path / "audit.jsonl"),
    )
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        response = client.post(
            "/api/models/compile",
            json=studio_request("PerfectIntegratorNeuron"),
            headers={"Origin": "http://127.0.0.1:5173"},
        )
        assert response.status_code == 200, response.text


def test_a_compile_the_compiler_accepts_still_runs(tmp_path: Path) -> None:
    """A model the compiler can build is not stopped by the check.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test-owned jobs and audit location.
    """
    settings = StudioRuntimeSettings(
        job_root_path=str(tmp_path / "jobs"),
        audit_log_path=str(tmp_path / "audit.jsonl"),
    )
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        response = client.post(
            "/api/models/compile",
            json=studio_request("PerfectIntegratorNeuron"),
            headers={"Origin": "http://127.0.0.1:5173"},
        )
        assert response.status_code == 200, response.text
        assert "module sc_" in response.json()["verilog"]


def test_a_map_at_its_own_step_compiles(tmp_path: Path) -> None:
    """A map sent its own step, as the Studio sends it, compiles; only another step is refused.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test-owned jobs and audit location.
    """
    settings = StudioRuntimeSettings(
        job_root_path=str(tmp_path / "jobs"),
        audit_log_path=str(tmp_path / "audit.jsonl"),
    )
    request = studio_request("RulkovMapNeuron")
    assert request["dt"] == 1.0
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        response = client.post(
            "/api/models/compile", json=request, headers={"Origin": "http://127.0.0.1:5173"}
        )
        assert response.status_code == 200, response.text
        assert "module sc_" in response.json()["verilog"]


def test_a_schema_without_a_step_is_a_server_fault(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no step in the request, the schema or the catalogue, the fault is the server's, not the caller's.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Gives the resolver a schema whose integration states no step.
    """
    from sc_neurocore.neurons.universal_dsl import load_schema

    detail = get_model_detail("SCLapicqueLIFNeuron")
    assert detail is not None
    stepless = {**load_schema("sc_lapicque_lif"), "integration": {"method": "exp_euler"}}
    monkeypatch.setattr(
        "sc_neurocore.studio.model_compile_configuration.load_schema", lambda _name: stepless
    )
    without_step = {key: value for key, value in detail.items() if key != "dt"}
    with pytest.raises(ValueError) as fault:
        resolve_model_compile_configuration(
            {"model_name": "SCLapicqueLIFNeuron", "q_format": "Q16.16"},
            detail_getter=lambda _name: without_step,
        )
    assert str(fault.value) == (
        "Studio model configuration field 'schema dt' must be a positive number."
    )
    assert not isinstance(fault.value, AuthoredRefusal)


def test_a_configuration_without_a_schema_name_is_a_server_fault() -> None:
    """A compile configuration that names no schema is the server's catalogue at fault, not the caller."""
    detail = get_model_detail("SCLapicqueLIFNeuron")
    assert detail is not None
    configuration = {
        key: value for key, value in detail["compile_configuration"].items() if key != "schema_name"
    }
    nameless = {**detail, "compile_configuration": configuration}
    with pytest.raises(ValueError) as fault:
        resolve_model_compile_configuration(
            {"model_name": "SCLapicqueLIFNeuron", "q_format": "Q16.16"},
            detail_getter=lambda _name: nameless,
        )
    assert str(fault.value) == (
        "Studio model configuration field 'schema_name' must be a non-empty string."
    )
    assert not isinstance(fault.value, AuthoredRefusal)
