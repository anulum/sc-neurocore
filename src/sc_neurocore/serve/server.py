# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Streaming SNN inference server

"""Real-time SNN inference server over HTTP/JSON.

Accepts spike events via POST requests, runs the SNN network for one
timestep per event batch, and returns output spike events. Designed
for real-time applications: robotics, BCI, event cameras.

Usage:
    sc-neurocore serve model.nir --port 8001

    # Client sends:
    POST /step {"inputs": {"input": [1.0, 0.0, 0.5]}}

    # Server returns:
    {"outputs": {"output": [0.0, 1.0]}, "timestep": 42}
"""

from __future__ import annotations

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import numpy as np

from sc_neurocore.refusals import AuthoredRefusal

DEFAULT_MAX_BODY_BYTES = 1 << 20
"""The largest ``/step`` request body the server reads, in bytes, unless a server is given another limit."""

STEP_FAILED = "The network step failed."
"""The fixed answer to any fault inside a step that is not a deliberate refusal."""


class ServeRefusal(AuthoredRefusal):
    """A deliberate refusal of the inference server, with text written for the caller."""


def _inputs_from(data: object) -> dict[str, list[float]]:
    """Return the ``inputs`` of a decoded ``/step`` body, or refuse a malformed one.

    Parameters
    ----------
    data : object
        The decoded JSON body.

    Returns
    -------
    dict of str to list of float
        Each node name with its finite numeric values.

    Raises
    ------
    ServeRefusal
        If the body is not an object, ``inputs`` is not an object of node names, or a value is not a
        list of finite numbers.
    """
    if not isinstance(data, dict):
        raise ServeRefusal("The request body must be a JSON object.")
    inputs = data.get("inputs", {})
    if not isinstance(inputs, dict):
        raise ServeRefusal("Field 'inputs' must map node names to arrays of numbers.")
    checked: dict[str, list[float]] = {}
    for name, values in inputs.items():
        if not isinstance(values, list) or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            for value in values
        ):
            raise ServeRefusal(f"Input {name!r} must be an array of finite numbers.")
        checked[name] = [float(value) for value in values]
    return checked


class SpikeServer:
    """Streaming SNN inference server.

    Parameters
    ----------
    network : Any
        The SNN to run: an object with ``step``, or a population network without projections
        whose populations carry unique non-empty string labels.
    host : str, default='127.0.0.1'
        Bind address.
    port : int, default=8001
        Listen port.
    max_body_bytes : int, default=DEFAULT_MAX_BODY_BYTES
        The largest ``/step`` request body read; a longer one is refused with 413 unread.

    Raises
    ------
    ServeRefusal
        If the network has projections without ``step``, if any served population lacks a unique
        non-empty string label, or if ``max_body_bytes`` is not a positive integer.
    """

    def __init__(
        self,
        network: Any,
        host: str = "127.0.0.1",
        port: int = 8001,
        *,
        max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
    ) -> None:
        """Validate the served network and configure the HTTP listener state."""
        if not hasattr(network, "step") and hasattr(network, "populations"):
            if getattr(network, "projections", None):
                raise ServeRefusal(
                    "A network with projections needs a step method; stepping its populations "
                    "alone would ignore every projection."
                )
            seen_labels: set[str] = set()
            for pop in network.populations:
                label = getattr(pop, "label", None)
                if not isinstance(label, str) or not label.strip():
                    raise ServeRefusal("Every served population requires a non-empty string label.")
                if label in seen_labels:
                    raise ServeRefusal(
                        f"Population label {label!r} is duplicated; every served population must "
                        "have a unique label."
                    )
                seen_labels.add(label)
        if (
            isinstance(max_body_bytes, bool)
            or not isinstance(max_body_bytes, int)
            or max_body_bytes < 1
        ):
            raise ServeRefusal("max_body_bytes must be a positive integer.")
        self.network = network
        self.host = host
        self.port = port
        self.max_body_bytes = max_body_bytes
        self._timestep = 0
        self._lock = threading.Lock()
        self._server: HTTPServer | None = None

    def reset(self) -> None:
        """Reset the network and the timestep, under the same lock as :meth:`step`."""
        with self._lock:
            self._timestep = 0
            if hasattr(self.network, "reset"):
                self.network.reset()

    def step(self, inputs: dict[str, list[float]]) -> dict[str, Any]:
        """Validate all population inputs before stepping any population and return output spikes.

        Parameters
        ----------
        inputs : dict of str to list of float
            Mapping from input node or population label to finite current values.

        Returns
        -------
        dict of str to Any
            Dictionary with ``'outputs'`` (node or population label mapped to spike values) and
            ``'timestep'``.

        Raises
        ------
        ServeRefusal
            If ``inputs`` names an unknown population, provides non-finite, boolean, or complex
            values, or supplies a vector whose length differs from ``pop.n``.
        TypeError
            If ``self.network`` exposes neither ``step`` nor ``populations``.
        """
        with self._lock:
            # SCNetwork (from NIR bridge)
            if hasattr(self.network, "step"):
                inp = {k: np.array(v) for k, v in inputs.items()}
                out = self.network.step(inp)
                self._timestep += 1
                return {
                    "outputs": {
                        k: v.tolist() if hasattr(v, "tolist") else v for k, v in out.items()
                    },
                    "timestep": self._timestep,
                }

            # Population-Projection Network (without projections)
            if hasattr(self.network, "populations"):
                populations = list(self.network.populations)
                by_label = {pop.label: pop for pop in populations}
                for name in inputs:
                    if name not in by_label:
                        raise ServeRefusal(f"Unknown input population {name!r}.")

                prepared: dict[str, np.ndarray[Any, Any]] = {}
                for pop in populations:
                    if pop.label in inputs:
                        raw_values = inputs[pop.label]
                        if not isinstance(raw_values, (list, tuple, np.ndarray)):
                            raise ServeRefusal(
                                f"Input {pop.label!r} must be an array of finite numbers."
                            )
                        if len(raw_values) != pop.n:
                            raise ServeRefusal(
                                f"Input {pop.label!r} requires {pop.n} values, "
                                f"got {len(raw_values)}."
                            )
                        arr = np.empty(pop.n, dtype=np.float64)
                        for idx, val in enumerate(raw_values):
                            if (
                                isinstance(val, bool)
                                or not isinstance(val, (int, float, np.integer, np.floating))
                                or not math.isfinite(float(val))
                            ):
                                raise ServeRefusal(
                                    f"Input {pop.label!r} must be an array of finite numbers."
                                )
                            arr[idx] = float(val)
                        prepared[pop.label] = arr
                    else:
                        prepared[pop.label] = np.zeros(pop.n, dtype=np.float64)

                results: dict[str, Any] = {}
                for pop in populations:
                    spikes = pop.step_all(prepared[pop.label])
                    results[pop.label] = spikes.tolist()
                self._timestep += 1
                return {"outputs": results, "timestep": self._timestep}

            raise TypeError(f"Unsupported network type: {type(self.network).__name__}")

    def start(self, blocking: bool = True) -> None:
        """Start the HTTP server.

        Parameters
        ----------
        blocking : bool, default=True
            If True, blocks until the server is shut down; if False, serves in a daemon thread.
        """
        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def _body(self) -> bytes | None:
                """Read the request body within the server's limit, or answer 411/400/413."""
                declared = self.headers.get("Content-Length")
                if declared is None:
                    self._respond(411, {"error": "Content-Length is required."})
                    return None
                if not declared.isdigit():
                    self._respond(400, {"error": "Content-Length must be a non-negative integer."})
                    return None
                length = int(declared)
                if length > server_ref.max_body_bytes:
                    self.close_connection = True
                    self._respond(
                        413,
                        {"error": f"The request body exceeds {server_ref.max_body_bytes} bytes."},
                    )
                    return None
                return self.rfile.read(length)

            def do_POST(self) -> None:
                if self.path == "/step":
                    body = self._body()
                    if body is None:
                        return
                    try:
                        inputs = _inputs_from(json.loads(body))
                    except ServeRefusal as refusal:
                        self._respond(400, {"error": str(refusal)})
                        return
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        self._respond(400, {"error": "The request body is not valid JSON."})
                        return
                    try:
                        result = server_ref.step(inputs)
                    except ServeRefusal as refusal:
                        self._respond(400, {"error": str(refusal)})
                        return
                    # Every other fault gets the fixed answer; its text stays inside the server.
                    except Exception:
                        self._respond(500, {"error": STEP_FAILED})
                        return
                    self._respond(200, result)
                elif self.path == "/reset":
                    server_ref.reset()
                    self._respond(200, {"status": "reset", "timestep": 0})
                elif self.path == "/info":
                    self._respond(
                        200,
                        {
                            "timestep": server_ref._timestep,
                            "type": type(server_ref.network).__name__,
                        },
                    )
                else:
                    self._respond(404, {"error": "Not found. Use /step, /reset, /info"})

            def do_GET(self) -> None:
                if self.path == "/info":
                    self._respond(
                        200,
                        {
                            "timestep": server_ref._timestep,
                            "type": type(server_ref.network).__name__,
                        },
                    )
                elif self.path == "/health":
                    self._respond(200, {"status": "ok"})
                else:
                    self._respond(404, {"error": "Not found"})

            def _respond(self, code: int, data: dict[str, Any]) -> None:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode("utf-8"))

            def log_message(self, format: str, *args: Any) -> None:
                pass  # suppress default logging

        self._server = HTTPServer((self.host, self.port), Handler)

        if blocking:
            print(f"SC-NeuroCore inference server on {self.host}:{self.port}")
            print("Endpoints: POST /step, POST /reset, GET /info, GET /health")
            self._server.serve_forever()
        else:
            thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            thread.start()

    def stop(self) -> None:
        """Shut down the HTTP server if running."""
        if self._server:
            self._server.shutdown()
