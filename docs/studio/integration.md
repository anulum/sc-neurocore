# Integration & Project Management

Block 6 ties all Studio components together: project save/load for
persistent workspaces, and a full pipeline that chains network design
through compilation to FPGA synthesis in a single action.

## Project Save/Load

Save and restore complete Studio state — equations, parameters, network
graph, synthesis target, training config — as JSON files on the server.

### Save

Click **Save** in the Projects panel (left sidebar), enter a name. The
full state is serialised into the server-side Studio project workspace. The
API response is path-free and returns `studio.project-save.v1` metadata with
SHA-256 digests for the project state and full saved project payload.
The Projects panel displays the returned evidence classification, project
name, state digest, project digest, and schema version after a successful save.

### Load

Click a project name to restore all state: equations, model selection,
parameters, network graph populations and projections, synthesis target, and
the Training Monitor configuration. Positive timebase settings and finite
numeric parameters are restored from saved values, including zero current,
while malformed optional fields fall back to Studio defaults before the next
simulation refresh.

### Storage Format

```json
{
  "name": "my_network",
  "saved_at": 1711504200.0,
  "version": "0.3.0",
  "state": {
    "sourceMode": "model",
    "selectedModelName": "LIFNeuron",
    "graphPopulations": [...],
    "graphProjections": [...],
    "synthTarget": "ice40",
    "trainingConfig": {...}
  }
}
```

## Full Pipeline

The pipeline runs from the Network Canvas with a single click:

```
Network Graph → Validate → Simulate → Compile → Synthesise
```

### Steps

1. **Validate** — check graph structure (populations exist, projections
   point to valid nodes, neuron count within 2000 limit)
2. **Simulate** — run the E-I balanced network simulation
3. **Compile** — generate SystemVerilog from LIF equations via the
   equation compiler
4. **Synthesise** — run Yosys on the generated Verilog for the selected
   FPGA target (ice40, ECP5, Gowin, Xilinx)

If any step fails, the pipeline stops and reports which step failed
and why.

### Using the Pipeline

1. Design your network on the Canvas tab
2. Select FPGA target in the Synthesis Dashboard (or use default ice40)
3. Click **Pipeline → ICE40** (or whichever target) on the Canvas toolbar
4. View results: simulation stats + synthesis resource usage

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/project/save` | Save project state |
| GET | `/api/project/list` | List all saved projects |
| GET | `/api/project/load/{name}` | Load a saved project |
| DELETE | `/api/project/{name}` | Delete a saved project |
| POST | `/api/pipeline/run` | Run full pipeline |

`/api/pipeline/run` executes through the Studio local worker manager. It keeps
the synchronous pipeline response used by the Network Canvas and also records a
`studio-pipeline` Admin queue job with the path-free result artifact
`pipeline/result.json`. The Network Canvas displays the same action-evidence
contract beside the terminal pipeline result, including evidence
classification, action kind, status, target, step, replay route, and the
`pipeline/result.json` plus `pipeline/evidence.json` artifact names.

### POST /api/project/save

```json
{"name": "my_network", "state": {"sourceMode": "model", ...}}
```

Returns path-free save evidence:

```json
{
  "evidence_classification": "project_workspace",
  "name": "my_network",
  "project_sha256": "<64 lowercase hex characters>",
  "saved_at": 1711504200.0,
  "schema_version": "studio.project-save.v1",
  "state_sha256": "<64 lowercase hex characters>",
  "status": "completed",
  "version": "0.3.0"
}
```

The frontend stores this response and renders it as the latest project-save
evidence strip in the Projects panel. The strip uses digest labels only; it
does not render server filesystem paths.

### POST /api/pipeline/run

```json
{
  "graph": {
    "populations": [...],
    "projections": [...],
    "duration": 200.0
  },
  "target": "ice40"
}
```

Returns:

```json
{
  "success": true,
  "target": "ice40",
  "pipeline": "graph → simulate → compile → synthesise",
  "steps": {
    "validate": {"passed": true},
    "simulate": {"n_spikes": 342, "n_total": 100},
    "compile": {"chars": 1580, "module": "sc_pipeline_neuron"},
    "synthesise": {"success": true, "target": "ice40", "resources": {...}}
  }
}
```
