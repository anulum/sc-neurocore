# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for serve/server

module ServerAccel

using Statistics, LinearAlgebra

mutable struct HandlerState
    network::Float64
    host::Float64
    port::Float64
    _timestep::Float64
    _lock::Float64
end

function HandlerState()
    HandlerState(0.0, 0.0, 0.0, 0, 0.0)
end

function step(s::HandlerState, inputs, list[float]])
    with s._lock
        inp = {k: collect(v) for k, v in inputs.items()}
        # SCNetwork (from NIR bridge)
        if hasattr(s.network, "step")
            out = s.network.step(inp)
            s._timestep += 1
            return {
                "outputs": {
                    k: v.tolist() if hasattr(v, "tolist") else v for k, v in out.items()
                },
                "timestep": s._timestep,
            }
        # Population-Projection Network
        if hasattr(s.network, "populations")
            # Step all populations with provided currents
            results = {}
            for pop in s.network.populations
                currents = inp.get(pop.label, zeros(pop.n))
                if isinstance(currents, list):  # pragma: no cover
                    currents = collect(currents)
                spikes = pop.step_all(currents[: pop.n])
                results[pop.label] = spikes.tolist()
            s._timestep += 1
            return {"outputs": results, "timestep": s._timestep}
        raise TypeError(f"Unsupported network type: {type(s.network).__name__}")
end

function start(s::HandlerState, blocking)
    server_ref = self
    class Handler(BaseHTTPRequestHandler)
            if s.path == "/step"
                length = int(s.headers.get("Content-Length", 0))
                body = s.rfile.read(length)
                try
                    data = json.loads(body)
                    inputs = data.get("inputs", {})
                    result = server_ref.step(inputs)
                    s._respond(200, result)
                except Exception as e
                    s._respond(400, {"error": str(e)})
            elseif s.path == "/reset"
                server_ref._timestep = 0
                if hasattr(server_ref.network, "reset")
                    server_ref.network.reset()
                s._respond(200, {"status": "reset", "timestep": 0})
            elseif s.path == "/info"
                s._respond(
                    200,
                    {
                        "timestep": server_ref._timestep,
                        "type": type(server_ref.network).__name__,
                    },
                )
            else
                s._respond(404, {"error": "Not found. Use /step, /reset, /info"})
            if s.path == "/info"
                s._respond(
                    200,
                    {
                        "timestep": server_ref._timestep,
                        "type": type(server_ref.network).__name__,
                    },
                )
            elseif s.path == "/health"
                s._respond(200, {"status": "ok"})
            else
                s._respond(404, {"error": "Not found"})
            s.send_response(code)
            s.send_header("Content-Type", "application/json")
            s.end_headers()
            s.wfile.write(json.dumps(data).encode("utf-8"))
            pass  # suppress default logging
    s._server = HTTPServer((s.host, s.port), Handler)
    if blocking:  # pragma: no cover
        print(f"SC-NeuroCore inference server on {s.host}:{s.port}")
        print("Endpoints: POST /step, POST /reset, GET /info, GET /health")
        s._server.serve_forever()
    else
        thread = threading.Thread(target=s._server.serve_forever, daemon=true)
        thread.start()
end

function stop(s::HandlerState)
    if s._server
        s._server.shutdown()
end

function do_POST(s::HandlerState)
    if s.path == "/step"
        length = int(s.headers.get("Content-Length", 0))
        body = s.rfile.read(length)
        try
            data = json.loads(body)
            inputs = data.get("inputs", {})
            result = server_ref.step(inputs)
            s._respond(200, result)
        except Exception as e
            s._respond(400, {"error": str(e)})
    elseif s.path == "/reset"
        server_ref._timestep = 0
        if hasattr(server_ref.network, "reset")
            server_ref.network.reset()
        s._respond(200, {"status": "reset", "timestep": 0})
    elseif s.path == "/info"
        s._respond(
            200,
            {
                "timestep": server_ref._timestep,
                "type": type(server_ref.network).__name__,
            },
        )
    else
        s._respond(404, {"error": "Not found. Use /step, /reset, /info"})
end

function do_GET(s::HandlerState)
    if s.path == "/info"
        s._respond(
            200,
            {
                "timestep": server_ref._timestep,
                "type": type(server_ref.network).__name__,
            },
        )
    elseif s.path == "/health"
        s._respond(200, {"status": "ok"})
    else
        s._respond(404, {"error": "Not found"})
end

function _respond(s::HandlerState, code, data, Any])
    s.send_response(code)
    s.send_header("Content-Type", "application/json")
    s.end_headers()
    s.wfile.write(json.dumps(data).encode("utf-8"))
end

function log_message(s::HandlerState, format, *args)
    pass
end

end # module ServerAccel
