# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Pre-step HTTP 400 input validation and population label guards for SpikeServer

"""Verify SpikeServer population label uniqueness and pre-step 400 validation on real networks."""

from __future__ import annotations

import http.client
import json
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from sc_neurocore.network import Network, Population
from sc_neurocore.serve.server import ServeRefusal, SpikeServer


def _build_two_population_network() -> tuple[Network, Population, Population]:
    """Construct a real two-population LIF network without projections."""
    exc = Population("SCLapicqueLIFNeuron", 3, label="exc")
    inh = Population("SCLapicqueLIFNeuron", 2, label="inh")
    net = Network(exc, inh)
    return net, exc, inh


def _voltages(pop: Population) -> list[float]:
    """Snapshot membrane voltages of every neuron in ``pop``."""
    return [float(neuron.v) for neuron in pop.neurons]


def _http_post_step(server: SpikeServer, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Send a real HTTP POST /step request to ``server`` and decode the JSON response."""
    assert server._server is not None
    body = json.dumps(payload).encode("utf-8")
    conn = http.client.HTTPConnection("127.0.0.1", server._server.server_address[1], timeout=10)
    try:
        conn.request(
            "POST",
            "/step",
            body=body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
        )
        resp = conn.getresponse()
        decoded: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        return resp.status, decoded
    finally:
        conn.close()


@pytest.fixture()
def live_pop_server() -> Iterator[tuple[SpikeServer, Population, Population]]:
    """Run SpikeServer on an ephemeral localhost port backed by real LIF populations."""
    net, exc, inh = _build_two_population_network()
    server = SpikeServer(net, host="127.0.0.1", port=0)
    server.start(blocking=False)
    try:
        yield server, exc, inh
    finally:
        server.stop()


@pytest.mark.parametrize("bad_label", [None, "", "   "])
def test_spike_server_refuses_unlabelled_or_blank_population(bad_label: Any) -> None:
    """Populations whose label is None, empty, or whitespace are refused at construction."""
    pop = Population("SCLapicqueLIFNeuron", 2, label="exc")
    pop.label = bad_label
    net = Network(pop)
    with pytest.raises(ServeRefusal, match="non-empty string label"):
        SpikeServer(net)


def test_spike_server_refuses_duplicate_population_labels() -> None:
    """Explicitly or implicitly colliding population labels are refused at construction."""
    p1 = Population("SCLapicqueLIFNeuron", 2, label="shared")
    p2 = Population("SCLapicqueLIFNeuron", 3, label="shared")
    with pytest.raises(ServeRefusal, match="duplicated"):
        SpikeServer(Network(p1, p2))

    default_a = Population("SCLapicqueLIFNeuron", 2)
    default_b = Population("SCLapicqueLIFNeuron", 3)
    with pytest.raises(ServeRefusal, match="duplicated"):
        SpikeServer(Network(default_a, default_b))


def test_http_short_second_population_returns_400_atomically_without_mutating_first(
    live_pop_server: tuple[SpikeServer, Population, Population],
) -> None:
    """Short input on the second population returns HTTP 400 and leaves the first unmutated."""
    server, exc, inh = live_pop_server
    exc_before = _voltages(exc)
    inh_before = _voltages(inh)

    status, answer = _http_post_step(
        server,
        {"inputs": {"exc": [5.0, 5.0, 5.0], "inh": [5.0]}},
    )

    assert status == 400
    assert "inh" in answer["error"]
    assert "requires 2 values, got 1" in answer["error"]
    assert server._timestep == 0
    assert _voltages(exc) == exc_before
    assert _voltages(inh) == inh_before


def test_http_oversized_population_input_returns_400_instead_of_silent_truncation(
    live_pop_server: tuple[SpikeServer, Population, Population],
) -> None:
    """Oversized input arrays are refused with HTTP 400 rather than truncated."""
    server, exc, _inh = live_pop_server
    exc_before = _voltages(exc)

    status, answer = _http_post_step(
        server,
        {"inputs": {"exc": [1.0, 2.0, 3.0, 4.0]}},
    )

    assert status == 400
    assert "requires 3 values, got 4" in answer["error"]
    assert server._timestep == 0
    assert _voltages(exc) == exc_before


def test_http_unknown_population_key_returns_400_atomically(
    live_pop_server: tuple[SpikeServer, Population, Population],
) -> None:
    """Unknown population keys in inputs are refused with HTTP 400 before stepping."""
    server, exc, inh = live_pop_server
    exc_before = _voltages(exc)
    inh_before = _voltages(inh)

    status, answer = _http_post_step(
        server,
        {"inputs": {"exc": [3.0, 3.0, 3.0], "nonexistent": [1.0, 1.0]}},
    )

    assert status == 400
    assert "Unknown input population 'nonexistent'" in answer["error"]
    assert server._timestep == 0
    assert _voltages(exc) == exc_before
    assert _voltages(inh) == inh_before


def test_direct_step_validates_nonfinite_boolean_complex_and_nonsequence_population_inputs() -> (
    None
):
    """Direct SpikeServer.step calls refuse non-finite, boolean, complex, and scalar inputs."""
    net, exc, _inh = _build_two_population_network()
    server = SpikeServer(net)
    exc_before = _voltages(exc)

    with pytest.raises(ServeRefusal, match="finite numbers"):
        server.step({"exc": [1.0, float("nan"), 0.0]})
    with pytest.raises(ServeRefusal, match="finite numbers"):
        server.step({"exc": [True, 0.0, 0.0]})
    np_bool_inputs: Any = {"exc": [np.bool_(True), 0.0, 0.0]}
    with pytest.raises(ServeRefusal, match="finite numbers"):
        server.step(np_bool_inputs)
    complex_inputs: Any = {"exc": [1.0 + 0.0j, 0.0, 0.0]}
    with pytest.raises(ServeRefusal, match="finite numbers"):
        server.step(complex_inputs)
    np_complex_inputs: Any = {"exc": [np.complex128(1.0 + 2.0j), 0.0, 0.0]}
    with pytest.raises(ServeRefusal, match="finite numbers"):
        server.step(np_complex_inputs)
    bad_scalar: Any = 5.0
    with pytest.raises(ServeRefusal, match="array of finite numbers"):
        server.step({"exc": bad_scalar})

    assert server._timestep == 0
    assert _voltages(exc) == exc_before

    valid_np_scalars: Any = {"exc": [np.float32(2.0), np.int64(0), np.float64(2.0)]}
    step_out = server.step(valid_np_scalars)
    assert step_out["timestep"] == 1
    assert len(step_out["outputs"]["exc"]) == 3

    # Calling stop() before start() and reset() on an object without reset() are safe no-ops.
    server.stop()
    bare_server = SpikeServer(object())
    bare_server.reset()


def test_http_valid_partial_and_full_population_steps_succeed_after_refusal(
    live_pop_server: tuple[SpikeServer, Population, Population],
) -> None:
    """Omitted populations default to zero drive and valid steps advance cleanly after a 400."""
    server, exc, _inh = live_pop_server

    bad_status, _ = _http_post_step(server, {"inputs": {"inh": [1.0]}})
    assert bad_status == 400

    ok_status, ok_body = _http_post_step(server, {"inputs": {"exc": [2.5, 0.0, 2.5]}})
    assert ok_status == 200
    assert ok_body["timestep"] == 1
    assert len(ok_body["outputs"]["exc"]) == 3
    assert ok_body["outputs"]["inh"] == [0, 0]
    assert not np.allclose(_voltages(exc), [-65.0, -65.0, -65.0])
