# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — inference server refusals over the real HTTP surface

"""The server answers only authored refusals or fixed sentences, bounds the body, and locks a reset."""

from __future__ import annotations

import http.client
import json
import threading
from collections.abc import Iterator
from typing import Any

import pytest

from sc_neurocore.serve.server import STEP_FAILED, ServeRefusal, SpikeServer

SECRET = "internal path /opt/model/weights.bin and a traceback line"


class _Network:
    """A step network whose behaviour the test chooses per request."""

    def __init__(self) -> None:
        self.fault: BaseException | None = None
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.resets = 0

    def step(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Hold until released, then raise the chosen fault or echo the inputs."""
        self.entered.set()
        self.release.wait(10)
        if self.fault is not None:
            raise self.fault
        return {name: list(values) for name, values in inputs.items()}

    def reset(self) -> None:
        """Count resets."""
        self.resets += 1


@pytest.fixture()
def served() -> Iterator[tuple[SpikeServer, _Network]]:
    """Yield a running server on an ephemeral port with a small body limit."""
    network = _Network()
    server = SpikeServer(network, port=0, max_body_bytes=256)
    server.start(blocking=False)
    yield server, network
    server.stop()


def _post(
    server: SpikeServer, path: str, body: bytes, headers: dict[str, str] | None = None
) -> tuple[int, dict[str, Any]]:
    """Send one POST and return the status and the decoded JSON answer."""
    assert server._server is not None
    connection = http.client.HTTPConnection(
        "127.0.0.1", server._server.server_address[1], timeout=10
    )
    connection.putrequest("POST", path)
    for name, value in (
        headers if headers is not None else {"Content-Length": str(len(body))}
    ).items():
        connection.putheader(name, value)
    connection.endheaders()
    connection.send(body)
    response = connection.getresponse()
    answer = json.loads(response.read() or b"{}")
    connection.close()
    return response.status, answer


def test_a_fault_inside_the_step_never_reaches_the_caller(
    served: tuple[SpikeServer, _Network],
) -> None:
    """Any non-refusal fault answers 500 with the fixed sentence; its own text stays inside."""
    server, network = served
    network.fault = RuntimeError(SECRET)
    status, answer = _post(server, "/step", json.dumps({"inputs": {"a": [1.0]}}).encode())
    assert (status, answer) == (500, {"error": STEP_FAILED})
    assert SECRET not in json.dumps(answer)


def test_an_authored_refusal_reaches_the_caller_with_its_text(
    served: tuple[SpikeServer, _Network],
) -> None:
    """A ServeRefusal raised by the step is the one message a caller may read."""
    server, network = served
    network.fault = ServeRefusal("Input 'a' needs three values.")
    assert _post(server, "/step", json.dumps({"inputs": {"a": [1.0]}}).encode()) == (
        400,
        {"error": "Input 'a' needs three values."},
    )


@pytest.mark.parametrize(
    ("body", "error"),
    [
        (b"not json", "The request body is not valid JSON."),
        (b"\xff\xfe", "The request body is not valid JSON."),
        (b"[1, 2]", "The request body must be a JSON object."),
        (b'{"inputs": [1, 2]}', "Field 'inputs' must map node names to arrays of numbers."),
        (b'{"inputs": {"a": "x"}}', "Input 'a' must be an array of finite numbers."),
        (b'{"inputs": {"a": [true]}}', "Input 'a' must be an array of finite numbers."),
        (b'{"inputs": {"a": [NaN]}}', "Input 'a' must be an array of finite numbers."),
    ],
)
def test_malformed_input_is_a_4xx_with_a_fixed_sentence(
    served: tuple[SpikeServer, _Network], body: bytes, error: str
) -> None:
    """Malformed input is refused with 400 and an authored sentence, never 500 or parser text."""
    server, _network = served
    assert _post(server, "/step", body) == (400, {"error": error})


def test_a_body_over_the_limit_is_refused_unread(served: tuple[SpikeServer, _Network]) -> None:
    """A declared length over the limit answers 413 without reading the body."""
    server, network = served
    status, answer = _post(server, "/step", b"", headers={"Content-Length": "257"})
    assert (status, answer) == (413, {"error": "The request body exceeds 256 bytes."})
    assert not network.entered.is_set()


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({}, (411, {"error": "Content-Length is required."})),
        (
            {"Content-Length": "-1"},
            (400, {"error": "Content-Length must be a non-negative integer."}),
        ),
        (
            {"Content-Length": "ten"},
            (400, {"error": "Content-Length must be a non-negative integer."}),
        ),
    ],
)
def test_a_missing_or_malformed_length_is_refused(
    served: tuple[SpikeServer, _Network],
    headers: dict[str, str],
    expected: tuple[int, dict[str, str]],
) -> None:
    """The body length must be declared as a plain non-negative integer."""
    server, _network = served
    assert _post(server, "/step", b"", headers=headers) == expected


def test_a_reset_waits_for_the_step_in_progress(served: tuple[SpikeServer, _Network]) -> None:
    """/reset takes the step lock, so it never interleaves with a running step."""
    server, network = served
    network.release.clear()
    stepping = threading.Thread(
        target=_post, args=(server, "/step", json.dumps({"inputs": {"a": [1.0]}}).encode())
    )
    stepping.start()
    assert network.entered.wait(10)
    resetting = threading.Thread(target=_post, args=(server, "/reset", b""))
    resetting.start()
    resetting.join(0.5)
    assert resetting.is_alive() and network.resets == 0
    network.release.set()
    stepping.join(10)
    resetting.join(10)
    assert network.resets == 1 and server._timestep == 0


class _Population:
    """A population with a label and a size."""

    def __init__(self) -> None:
        self.label = "exc"
        self.n = 2


class _ProjectedNetwork:
    """A population network with a projection and no step method."""

    def __init__(self) -> None:
        self.populations = [_Population()]
        self.projections = [object()]


def test_a_network_whose_projections_would_be_ignored_is_refused() -> None:
    """Stepping only the populations would drop every projection, so the server refuses the network."""
    with pytest.raises(ServeRefusal, match="needs a step method"):
        SpikeServer(_ProjectedNetwork())


def test_a_blocking_server_announces_itself_and_stops_on_request(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """In blocking mode the server prints its address and endpoints, serves, and returns once stopped."""
    server = SpikeServer(_Network(), port=0)
    serving = threading.Thread(target=server.start, kwargs={"blocking": True})
    serving.start()
    for _ in range(200):
        if server._server is not None:
            break
        threading.Event().wait(0.05)
    assert server._server is not None
    assert _post(server, "/step", json.dumps({"inputs": {"a": [2.0]}}).encode()) == (
        200,
        {"outputs": {"a": [2.0]}, "timestep": 1},
    )
    server.stop()
    serving.join(10)
    assert not serving.is_alive()
    printed = capsys.readouterr().out
    assert "SC-NeuroCore inference server on 127.0.0.1:0" in printed
    assert "Endpoints: POST /step, POST /reset, GET /info, GET /health" in printed


@pytest.mark.parametrize("limit", [0, -5, True, 1.5])
def test_the_body_limit_must_be_a_positive_integer(limit: object) -> None:
    """A limit that is not a positive integer is refused at construction."""
    wrong: Any = limit
    with pytest.raises(ServeRefusal, match="max_body_bytes"):
        SpikeServer(_Network(), max_body_bytes=wrong)
