# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — A co-simulation the Studio does not run says why, before any job runs

"""A co-simulation the Studio would not run is answered 422 with its reason, not a failed job.

Measured before this was written, on main ``451cb71ae``: a request for an
integrator the model offers no bit-exact co-simulation path for, and a
current its Q-format cannot hold, were each admitted as a job and answered
HTTP 500 ``studio_job_failed``, while the job ledger kept the reason. The
Studio itself never sends the first (its request builder refuses it), so only
a direct caller met it. Of the 65 catalogue models with a compile
configuration, 24 offer no co-simulation integrator at all, 31 offer ``map``
and 10 offer ``euler``.
"""

from pathlib import Path
import shutil

import pytest
from starlette.testclient import TestClient

from sc_neurocore.compiler.verilog_compiler_config import Q88
from sc_neurocore.refusals import AuthoredRefusal
from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.model_catalogue import get_model_detail
from sc_neurocore.studio.model_compile_configuration import resolve_model_compile_configuration
from sc_neurocore.studio.platform import StudioRuntimeSettings

HAS_COSIM_TOOLS = all(shutil.which(tool) is not None for tool in ("gcc", "iverilog", "vvp"))


def cosim_request(
    model_name: str,
    *,
    integrator: str | None = None,
    q_format: str | None = None,
    current: float = 0.5,
) -> dict[str, object]:
    """Build a co-simulation request as the Studio does, with a chosen integrator and current.

    Parameters
    ----------
    model_name : str
        A catalogue model with a compile configuration.
    integrator : str, optional
        The integrator asked for; None asks for the first one the model offers for
        co-simulation, or its default integrator when it offers none.
    q_format : str, optional
        The format asked for; None asks for the model's default.
    current : float, optional
        The constant input. Default 0.5.

    Returns
    -------
    dict[str, object]
        The catalogue defaults, the step, integrator and Q-format, the current, 16 steps.
    """
    detail = get_model_detail(model_name)
    assert detail is not None
    configuration = detail["compile_configuration"]
    offered = configuration["cosim_integrators"] or [configuration["default_integrator"]]
    return {
        "model_name": model_name,
        "dt": detail["dt"],
        "integrator": offered[0] if integrator is None else integrator,
        "q_format": configuration["default_q_format"] if q_format is None else q_format,
        "params": {parameter["name"]: parameter["default"] for parameter in detail["params"]},
        "current": current,
        "n_steps": 16,
    }


def _post(tmp_path: Path, body: dict[str, object]) -> tuple[int, object, int]:
    """Send one co-simulation request; return its status, detail and how many jobs it admitted."""
    settings = StudioRuntimeSettings(
        job_root_path=str(tmp_path / "jobs"),
        audit_log_path=str(tmp_path / "audit.jsonl"),
    )
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        before = client.get("/api/studio/jobs").json()
        response = client.post(
            "/api/models/cosim", json=body, headers={"Origin": "http://127.0.0.1:5173"}
        )
        after = client.get("/api/studio/jobs").json()
    admitted = len(after.get("jobs", [])) - len(before.get("jobs", []))
    return response.status_code, response.json().get("detail"), admitted


#: A model that offers no co-simulation integrator, asked for its own default, and one that
#: offers euler, asked for rk4, which it compiles with but cannot co-simulate.
INTEGRATORS = [
    ("SCScaledResetAdaptiveIFNeuron", "rk4", "it offers none"),
    ("AdExNeuron", "rk4", "it offers euler"),
]


@pytest.mark.parametrize(("model_name", "integrator", "offered"), INTEGRATORS)
def test_an_integrator_without_a_bit_exact_path_answers_422_and_runs_no_job(
    tmp_path: Path, model_name: str, integrator: str, offered: str
) -> None:
    """The caller is told the model has no bit-exact path for that integrator, and what it offers.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test-owned jobs and audit location.
    model_name : str
        A model whose compile configuration does not list the integrator for co-simulation.
    integrator : str
        The integrator asked for.
    offered : str
        How the refusal names what the model offers instead.
    """
    status, detail, admitted = _post(tmp_path, cosim_request(model_name, integrator=integrator))
    assert status == 422
    assert detail == (
        f"Integrator {integrator!r} has no bit-exact co-simulation path for {model_name!r}; "
        f"{offered}."
    )
    assert admitted == 0


@pytest.mark.parametrize("side", [-1, 1])
def test_a_current_its_format_cannot_hold_answers_422_and_runs_no_job(
    tmp_path: Path, side: int
) -> None:
    """A current outside the Q-format's range would wrap its input word; the caller is told so.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test-owned jobs and audit location.
    side : int
        Below the format's range (-1) or above it (1).
    """
    q = Q88(data_width=32, fraction=16)
    current = side * (q.max_value * 4 + 1.0)
    status, detail, admitted = _post(tmp_path, cosim_request("RulkovMapNeuron", current=current))
    assert status == 422
    assert detail == (
        f"Studio model co-simulation current {current!r} is outside Q16.16 "
        f"[{q.min_value!r}, {q.max_value!r}]; its input word would wrap."
    )
    assert admitted == 0


@pytest.mark.parametrize("end", ["min_value", "max_value"])
def test_both_ends_of_the_format_s_range_are_admitted(end: str) -> None:
    """The range is closed: a current at either end is co-simulated, not refused.

    Parameters
    ----------
    end : str
        Which end of the Q16.16 range the current sits at.
    """
    from sc_neurocore.studio.model_cosim import refuse_cosim_request

    configuration = resolve_model_compile_configuration(cosim_request("RulkovMapNeuron"))
    q = Q88(data_width=32, fraction=16)
    refuse_cosim_request(configuration, current=getattr(q, end), cosim_integrators=["map"])


def test_each_co_simulation_refusal_is_authored() -> None:
    """The request boundary answers 422 for exactly this class, so both refusals carry it."""
    from sc_neurocore.studio.model_cosim import ModelCosimRefused, refuse_cosim_request

    rk4 = resolve_model_compile_configuration(cosim_request("AdExNeuron", integrator="rk4"))
    with pytest.raises(ModelCosimRefused) as refused_integrator:
        refuse_cosim_request(rk4, current=0.5, cosim_integrators=["euler"])
    rulkov = resolve_model_compile_configuration(cosim_request("RulkovMapNeuron"))
    with pytest.raises(ModelCosimRefused) as refused_current:
        refuse_cosim_request(rulkov, current=1.0e9, cosim_integrators=["map"])
    for refusal in (refused_integrator.value, refused_current.value):
        assert isinstance(refusal, AuthoredRefusal)
        assert isinstance(refusal, ValueError)


def test_a_compile_refusal_still_answers_first(tmp_path: Path) -> None:
    """A request the compiler refuses is refused for that reason, before the co-simulation check.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test-owned jobs and audit location.
    """
    body = cosim_request("SCScaledResetAdaptiveIFNeuron", q_format="Q8.8")
    status, detail, admitted = _post(tmp_path, body)
    assert status == 422
    assert isinstance(detail, str) and detail.startswith("Q8.8 cannot hold this neuron")
    assert admitted == 0


def test_a_server_side_fault_in_the_check_is_left_to_the_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fault in the server's own catalogue is not the caller's input; the job records it.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test-owned jobs and audit location.
    monkeypatch : pytest.MonkeyPatch
        Makes the check's lookup of the offered integrators fail; the worker resolves on its own.
    """
    import sc_neurocore.studio.api.compiler as compiler_routes

    def faulty_offer(_model_name: str) -> list[str]:
        raise KeyError("cosim_integrators")

    monkeypatch.setattr(compiler_routes, "cosim_integrators_of", faulty_offer)
    status, _detail, admitted = _post(
        tmp_path, cosim_request("SCScaledResetAdaptiveIFNeuron", integrator="rk4")
    )
    assert status == 500
    assert admitted == 1


@pytest.mark.skipif(not HAS_COSIM_TOOLS, reason="GCC and Icarus Verilog are required")
def test_a_co_simulation_the_studio_offers_still_runs(tmp_path: Path) -> None:
    """A map at a current its format holds is co-simulated, bit-exact, as before the check.

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
            "/api/models/cosim",
            json=cosim_request("RulkovMapNeuron"),
            headers={"Origin": "http://127.0.0.1:5173"},
        )
    assert response.status_code == 200, response.text
    assert response.json()["bit_exact"] is True
