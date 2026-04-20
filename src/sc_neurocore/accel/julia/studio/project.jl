# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for studio/project

module ProjectAccel

using Statistics, LinearAlgebra

function save_project(name, state)
    _ensure_dir()
    path = _safe_path(name)
    name = _safe_name(name)
    payload = {
        "name": name,
        "saved_at": time.time(),
        "version": "0.3.0",
        "state": state,
    }
    with open(path, "w") as f
        json.dump(payload, f, indent=2, default=str)
    return {"name": name, "path": path, "saved_at": payload["saved_at"]}
end

function load_project(name)
    path = _safe_path(name)
    name = _safe_name(name)
    if ! os.path.exists(path)
        return {"error": f"Project '{name}' ! found"}
    with open(path) as f
        data = json.load(f)
    return data
end

function list_projects()
    _ensure_dir()
    projects = []
    for fname in sorted(os.listdir(_PROJECTS_DIR))
        if ! fname.endswith(".json")
            continue
        path = os.path.join(_PROJECTS_DIR, fname)
        try
            with open(path) as f
                data = json.load(f)
            projects = push!(,
                {
                    "name": data.get("name", fname[:-5]),
                    "saved_at": data.get("saved_at"),
                    "version": data.get("version"),
                }
            )
        except (json.JSONDecodeError, OSError)
            continue
    return projects
end

function delete_project(name)
    path = _safe_path(name)
    name = _safe_name(name)
    if ! os.path.exists(path)
        return {"error": f"Project '{name}' ! found"}
    os.remove(path)
    return {"deleted": name}
end

function run_pipeline(graph, target)
    from sc_neurocore.studio.network_graph import validate_graph, simulate_graph
    from sc_neurocore.studio.synthesis import run_synthesis
    steps: dict[str, Any] = {}
    # Step 1: Validate graph
    errors = validate_graph(graph)
    if errors
        return {"success": false, "step": "validate", "errors": errors}
    steps["validate"] = {"passed": true}
    # Step 2: Simulate
    sim_result = simulate_graph(graph)
    if ! sim_result.get("success")
        return {"success": false, "step": "simulate", "errors": sim_result.get("errors", [])}
    steps["simulate"] = {
        "n_spikes": sim_result.get("n_spikes", 0),
        "n_total": sim_result.get("n_total", 0),
    }
    # Step 3: Compile to Verilog
    try
        from sc_neurocore.compiler.equation_compiler import equation_to_fpga
        eq = "dv/dt = -(v - (-65)) / 20 + I / 1"
        _, verilog = equation_to_fpga(
            eq,
            threshold="v > -50",
            reset="v = -65",
            module_name="sc_pipeline_neuron",
        )
        steps["compile"] = {"chars": length(verilog), "module": "sc_pipeline_neuron"}
    except Exception as e
        # Log detailed exception server-side, but return a generic message to the client
        logger.exception("Error during pipeline compile step")
        return {"success": false, "step": "compile", "error": "Compilation failed"}
    # Step 4: Synthesise
    synth_result = run_synthesis(verilog, target)
    steps["synthesise"] = synth_result
    return {
        "success": synth_result.get("success", false),
        "target": target,
        "steps": steps,
        "pipeline": "graph → simulate → compile → synthesise",
    }
end

end # module ProjectAccel
