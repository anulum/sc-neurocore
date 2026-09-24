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

The pipeline builds hardware for the network drawn on the Canvas, or says
exactly why it cannot:

```
Network Graph → Validate → Simulate → Lower → Co-simulate → Synthesise
```

### Steps

1. **Validate** — check the graph (populations exist, projections point to
   valid nodes, neuron count within the 2000 limit).
2. **Simulate** — run the network in the Studio runtime.
3. **Lower** — translate the network for the hardware network compiler with
   each catalogue model's own step. Supported: `SCLapicqueLIFNeuron` (profile
   `sc_lif`, the exact step `v ← v·e^(−dt/τ) + (v_rest + R·I)(1 − e^(−dt/τ))`,
   firing at `v ≥ v_threshold`) and `PerfectIntegratorNeuron` (`r = 1/c_m`,
   with its profile's threshold comparison). Projections keep the connectivity
   the runtime realises and their delay in steps; the hardware's registered
   spikes are the runtime's one-step latency. A constant drive enters through
   one input lane per neuron. Refused before any hardware is generated, every
   reason at once: another model, a Poisson drive, a membrane that does not
   start at rest, a value outside the fixed-point range or one that quantises
   to zero, a delay beyond 1024 steps, and two populations of one model with
   different parameters. Values that round are listed in the step's `notes`.
4. **Co-simulate** — compile the RTL and run it in Icarus Verilog beside a C
   model built from the compiler's bit-true neuron kernels and the compiled
   interconnect, and beside the Studio's own run, for up to 2000 steps. The RTL
   must reproduce its model on every step, or synthesis does not run. Where the
   fixed-point hardware and the Studio's double-precision run differ, the first
   differing step is reported.
5. **Synthesise** — run Yosys on the top module and the neurons it instantiates
   for the selected FPGA target (ice40, ECP5, Gowin, Xilinx).

The result's `trace` binds the run: the lowering's `input_sha256` (the graph's
digest, the fixed-point format and the lowering schema), the RTL, the bit-true
model and the synthesised source. Two different networks never share them.
The format is `Q8.8` unless the request asks for `Q16.16`; exact values such as
0.375 reproduce the Studio run in either, while a leaky neuron's decay rounds
visibly in Q8.8.

Co-simulation needs `iverilog`, `vvp` and a C compiler on the server; without
them the pipeline stops at the co-simulation step and says which tool is
missing.

### Using the Pipeline

1. Design your network on the Canvas tab
2. Select FPGA target in the Synthesis Dashboard (or use default ice40)
3. Click **Pipeline → ICE40** (or whichever target) on the Canvas toolbar
4. Read the result under the canvas: the step it ended at, every refusal
   reason, and what the co-simulation established

## Hub Federation Manifest

The optional Hub-facing federation surface lives under
`sc_neurocore.federation` and is separate from this local Studio web app. It
declares the schema-A capability manifest, the eight advertised verbs, and the
evidence-bundle contracts consumed by SCPN Studio platform federation.

Install the optional SDK before generating or checking the committed manifest:

```bash
pip install "sc-neurocore[federation]"
python tools/emit_studio_manifest.py --check
```

The generated artifact is `docs/_generated/studio_manifest.json`; it stamps the
source package version and hashes the declared verbs plus evidence schemas, not
git state. API details and envelope examples are in
[Studio Federation API](../api/federation.md).

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/project/save` | Save project state |
| GET | `/api/project/list` | List all saved projects |
| GET | `/api/project/load/{name}` | Load a saved project |
| DELETE | `/api/project/{name}` | Delete a saved project |
| POST | `/api/pipeline/run` | Lower, co-simulate and synthesise the network, or refuse it |

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
    "duration": 30.0,
    "dt": 1.0
  },
  "target": "ice40",
  "q_format": "Q8.8"
}
```

Returns, for a supported network:

```json
{
  "success": true,
  "step": "synthesise",
  "target": "ice40",
  "pipeline": "graph → simulate → lower → co-simulate → synthesise",
  "steps": {
    "validate": {"passed": true},
    "simulate": {"n_spikes": 48, "n_total": 5},
    "lower": {"input_sha256": "…", "q_format": "Q8.8", "populations": [...], "notes": []},
    "cosimulate": {
      "steps": 30,
      "rtl_matches_bit_true_model": true,
      "studio_agreement": {"identical": true, "first_divergent_step": null},
      "spike_counts": {"rtl": 48, "bit_true_model": 48, "studio": 48}
    },
    "synthesise": {"success": true, "target": "ice40", "resources": {...}}
  },
  "trace": {
    "input_sha256": "…",
    "rtl_sha256": "…",
    "bit_true_model_sha256": "…",
    "synthesis_source_sha256": "…"
  }
}
```

A graph the lowering refuses stops at `"step": "lower"` with every reason in
`reasons`.
