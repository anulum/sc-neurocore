# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
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
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np


class SpikeServer:
    """Streaming SNN inference server.

    Parameters
    ----------
    network : SCNetwork or Network
        The SNN to run.
    host : str
        Bind address (default '0.0.0.0').
    port : int
        Listen port (default 8001).
    """

    def __init__(self, network, host: str = "0.0.0.0", port: int = 8001):
        self.network = network
        self.host = host
        self.port = port
        self._timestep = 0
        self._lock = threading.Lock()
        self._server = None

    def step(self, inputs: dict[str, list[float]]) -> dict:
        """Run one network timestep and return output spikes.

        Parameters
        ----------
        inputs : dict mapping input node name to value array

        Returns
        -------
        dict with 'outputs' (node->values) and 'timestep'
        """
        with self._lock:
            inp = {k: np.array(v) for k, v in inputs.items()}

            # SCNetwork (from NIR bridge)
            if hasattr(self.network, "step"):
                out = self.network.step(inp)
                self._timestep += 1
                return {
                    "outputs": {
                        k: v.tolist() if hasattr(v, "tolist") else v for k, v in out.items()
                    },
                    "timestep": self._timestep,
                }

            # Population-Projection Network
            if hasattr(self.network, "populations"):
                # Step all populations with provided currents
                results = {}
                for pop in self.network.populations:
                    currents = inp.get(pop.label, np.zeros(pop.n))
                    if isinstance(currents, list):
                        currents = np.array(currents)
                    spikes = pop.step_all(currents[: pop.n])
                    results[pop.label] = spikes.tolist()
                self._timestep += 1
                return {"outputs": results, "timestep": self._timestep}

            raise TypeError(f"Unsupported network type: {type(self.network).__name__}")

    def start(self, blocking: bool = True):
        """Start the HTTP server.

        Parameters
        ----------
        blocking : bool
            If True (default), blocks until server is shut down.
            If False, runs in a background thread.
        """
        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path == "/step":
                    length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(length)
                    try:
                        data = json.loads(body)
                        inputs = data.get("inputs", {})
                        result = server_ref.step(inputs)
                        self._respond(200, result)
                    except Exception as e:
                        self._respond(400, {"error": str(e)})
                elif self.path == "/reset":
                    server_ref._timestep = 0
                    if hasattr(server_ref.network, "reset"):
                        server_ref.network.reset()
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

            def do_GET(self):
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

            def _respond(self, code, data):
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode("utf-8"))

            def log_message(self, format, *args):
                pass  # suppress default logging

        self._server = HTTPServer((self.host, self.port), Handler)

        if blocking:
            print(f"SC-NeuroCore inference server on {self.host}:{self.port}")
            print("Endpoints: POST /step, POST /reset, GET /info, GET /health")
            self._server.serve_forever()
        else:
            thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            thread.start()

    def stop(self):
        """Shut down the server."""
        if self._server:
            self._server.shutdown()
