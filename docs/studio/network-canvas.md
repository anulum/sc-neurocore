# Network Canvas

The Network Canvas is a visual editor for small spiking networks: add
populations, connect them with projections, set their models, parameters,
inputs, weights and delays, then run the graph. Built on React Flow (Xyflow),
the canvas supports node dragging, edge creation via drag-connect and layout
updates.

A graph runs exactly what it declares. The server resolves the canvas JSON
into a versioned specification (`studio.network-graph-spec.v1`), lowers it to
the public `sc_neurocore.network` objects (`Population`, `Projection`,
`SpikeMonitor`, `StepCurrent`, `PoissonInput`) and runs the reference Python
loop. Nothing is mapped to a template, clamped, rounded or defaulted from a
different model: a graph the runtime cannot preserve is rejected with the
field and the reason.

## Quick Start

1. Switch to the **Canvas** tab
2. Click **+ Exc** to add an excitatory population (constant input `I = 1.2`)
3. Click **+ Inh** to add an inhibitory population (no external input)
4. Drag from one node's handle to another to create a projection
5. Click **Simulate** to run the graph
6. Read the per-population spike counts and rates in the results bar

## Editing the graph

Adding, connecting and moving are what the toolbar and the canvas suggest.
Deleting is worth stating exactly, because it changes the graph and not only
the picture:

- **Deleting a population also deletes every projection into or out of it.** A
  network cannot hold an edge whose endpoint is gone, so the edges leave with
  the node rather than being left behind as references nothing can resolve.
- **Deleting a projection leaves both populations in place.**
- **Moving a node changes the layout only.** A position travels with the graph
  so a saved workspace reopens as you left it, but the run resolves the same
  specification wherever the node sits: dragging never changes what is
  simulated.

**Undo** steps back through graph edits: `Ctrl+Z` (`Cmd+Z`), or the Undo
button; `Ctrl+Shift+Z` steps forward again. Adding, deleting and changing a
population or a projection are edits. **Moving a node is not** — a drag writes a
position on every frame, and recording those would bury the edits worth undoing
under layout noise.

The history holds the last 50 edits and lives for the session; it is not saved
with the workspace, so a reopened workspace starts with an empty history and
the graph exactly as you saved it. Editing after an undo drops the redo branch,
because a redo into a graph that no longer follows from the current one would
reinstate work you have already moved past.

## Editing a projection

Click a projection to open its property editor. Every field the runtime
executes is there — weight, rule, probability, delay, seed and autapses — and
each input states the contract it has to satisfy: the sign the source
population's type requires, that a delay is a whole number of graph timesteps
and is never rounded, that a probability lies in (0, 1], that an empty seed
means "derive one from the graph seed and the edge index". The statement is
tied to its input, so a screen reader reads it with the field rather than
leaving it to be found elsewhere on the page.

The editor **parses**; the server **decides**. A blank box or text where a
number belongs is refused in the browser, because it is not a value at all. A
weight whose sign contradicts its source, a delay that is not a whole step, a
probability out of range: those go to the server, which owns the contract and
answers with the field each failure came from. A second copy of the contract in
the browser would be a copy free to drift from the one that runs.

Each edit that parses is applied to the graph and validated, so the refusal
appears against the input that caused it, and disappears when it is fixed.
Switching the rule to `all_to_all` drops the probability rather than sending
one the specification refuses. Deleting a projection — or the population it
touches — closes the editor rather than leaving it writing into nothing.

## Reading the graph without the canvas

A node-and-edge diagram carries its meaning in positions and arrows, and
neither survives a screen reader: the canvas reports a list of draggable boxes
and says nothing about what is connected to what. The **Table view** toggle
states the topology instead of drawing it.

The table holds one row per population. The population is the row header, so a
screen reader announces which population a cell belongs to, and the row also
carries one sentence describing it in full — its model, count, type and input,
what reaches it and what it reaches, and what validation refused about it — for
a reader who does not want to walk the cells. A **Problems** column carries the
refusals themselves, per population and per connection, and reads `none` when
there are none. The caption states the size of the topology before you enter the
table. Each row's delete control says what it removes *and how many
projections leave with it*, because a column of controls all called "Delete" is
unusable without sight.

The table is a second presentation of the same graph the canvas draws and the
server runs, derived from the same fields: it is never a summary that could
drift from the graph. Deleting through it is the same edit as deleting on the
canvas, undo included. While the table is shown the canvas is hidden, so it
does not sit in the tab order behind it; the canvas keeps its viewport, and the
toggle returns you to it.

## Populations

Each population is a group of identical neurons of one catalogue model:

| Property | Default | Description |
|----------|---------|-------------|
| label | auto | Display name (e.g., "Exc 0") |
| model | `SCLapicqueLIFNeuron` | Catalogue class name; `GET /api/graph/models` lists the admissible ones |
| count | 80 (exc) / 20 (inh) | Positive integer |
| neuron_type | excitatory | `excitatory` or `inhibitory`; fixes the sign of outgoing weights |
| params | `{}` | Constructor overrides validated against the model's own contract |
| drive | `{"kind": "none"}` | External input: `none`, `constant` (`current`), or `poisson` (`rate_hz`, `weight`, optional `seed`) |
| position | auto | Canvas x, y coordinates |

The graph timestep `dt` is passed to every model constructor. A model that
cannot take it (fixed step attribute, no timestep field) rejects the graph.
Models with an integer drive or a `seed` constructor field are not admitted
(every neuron of a population would share the seed and its noise).

Excitatory populations are shown with rounded blue borders, inhibitory ones
with square red borders; the node shows model × count and the drive.

## Projections

Projections are directed connections between populations:

| Property | Default | Description |
|----------|---------|-------------|
| weight | +40 (exc source) / −40 (inh source) | Signed synaptic weight; the sign must agree with the source population's type |
| delay | 0 ms | Whole number of graph timesteps; a fractional delay is rejected, never rounded |
| rule | `random` | `random` (Erdős–Rényi with `probability`) or `all_to_all` (no probability) |
| probability | 0.2 | Connection probability of the `random` rule, in (0, 1] |
| seed | derived | Connectivity seed; derived from the graph seed and the edge index when absent |
| autapses | false | Self-projections drop the diagonal unless declared |

Two projections between the same pair are two independent projections whose
currents add. Populations without incoming projections or a drive stay
silent.

## Simulation

`POST /api/graph/simulate` takes the populations, projections, `duration`
(ms), `dt` (ms, default 0.1) and `seed` (default 42). The response is
`studio.network-graph-result.v1`:

- `spec`: the resolved specification with effective parameters, derived seeds,
  delays in steps and the `graph_sha256` digest;
- `execution`: the loop that ran (public `Network._run_python`), its step
  order, the one-step projection latency, the delay and synapse semantics, the
  `Network.run` timestep in seconds and the rejected Rust runner with its
  reason (default construction, no stimuli);
- `populations`: per population every spike event `(step, neuron)`, the mean
  rate and a binned rate;
- `topology`: per projection the synapse count, delay mode, autapses removed,
  the CSR digest and, within the element budget, the CSR arrays;
- `contract`: the metric contract of the reported activity;
- `n_total`, `n_spikes`, `spike_times` (ms), `spike_neurons` (global index,
  population offsets in `spec`), `graph_summary`.

Execution semantics of the public loop: a spike at step `t` reaches its
targets at step `t + 1 + delay_steps`; each source spike injects the weight
as drive into the target for one step, so the membrane increment per spike is
model-defined (about weight × dt / tau for the default LIF). Delays are in
milliseconds and must be whole steps.

**Limits:** at most 2000 neurons, 100000 steps and 1000000 neuron-steps per
synchronous run. A larger run is refused, not shortened.

A validation failure answers `200` with `success: false` and every message;
a run that fails numerically answers `422` with `graph_execution_failed`.

### Where a failure happened

`POST /api/graph/validate` answers with `valid`, the flat `errors` list of
messages it has always returned, and `issues` — the same failures, each with
the request field it came from:

```json
{
  "errors": ["Projection e1 delay 0.05 ms is not a whole number of 0.1 ms steps; …"],
  "issues": [
    {
      "field": "projections[0].delay",
      "message": "Projection e1 delay 0.05 ms is not a whole number of 0.1 ms steps; …"
    }
  ],
  "valid": false
}
```

The index is the position in the array that was **sent**, so a caller can place
each message against the object it holds. Fields are
`populations[i].<attribute>`, `projections[i].<attribute>` — including nested
ones such as `populations[0].params.tau` — or a bare `dt`, `duration`, `seed`
for a failure about the run as a whole.

The canvas resolves each field to the population or projection it names and
prefixes the message with it — `Exc 0 → Inh 0: …` — leaving the server's
sentence verbatim, and the table view puts it in that object's row. A message
whose field names an index the graph no longer holds, or a shape this build
does not recognise, is still shown, named by its own field: a refusal that
cannot be placed still has to be read.

## NIR Export/Import

The canvas exports and imports the NIR-named JSON format
(`format: "nir"`, `version: "0.1"`); this is a JSON interchange of the graph,
not a conformance proof against the NIR specification.

- **Export:** populations become nodes (`type` is the catalogue model name),
  projections become edges (weight, delay; the probability is not carried)
- **Import:** node `type` must be a catalogue model name (NIR primitives such
  as `LIF` are not mapped to a model); imported edges connect all-to-all. The
  assembled graph is validated with the default timestep and rejected when it
  would not execute.

```json
{
  "format": "nir",
  "version": "0.1",
  "nodes": {
    "pop_a": {"type": "SCLapicqueLIFNeuron", "count": 80, "neuron_type": "excitatory"},
    "pop_b": {"type": "SCLapicqueLIFNeuron", "count": 20, "neuron_type": "inhibitory"}
  },
  "edges": [
    {"source": "pop_a", "target": "pop_b", "weight": 40.0, "delay": 1.0}
  ]
}
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/graph/models` | List catalogue models admissible for populations |
| POST | `/api/graph/population` | Create a population node |
| POST | `/api/graph/projection` | Create a projection edge |
| POST | `/api/graph/validate` | Validate a graph; every error at once, each with the field it came from |
| POST | `/api/graph/simulate` | Run the graph through the public Network runtime |
| POST | `/api/graph/export-nir` | Export to the NIR-named JSON |
| POST | `/api/graph/import-nir` | Import from the NIR-named JSON |

### POST /api/graph/population

```json
{"label": "Exc 0", "model": "SCLapicqueLIFNeuron", "count": 80, "neuron_type": "excitatory",
 "x": 100, "y": 100, "params": {"tau": 10.0}, "drive": {"kind": "constant", "current": 1.2}}
```

### POST /api/graph/simulate

```json
{
  "populations": [...],
  "projections": [...],
  "duration": 200.0,
  "dt": 0.1,
  "seed": 42
}
```

## Supported operations

| Operation | Status |
|-----------|--------|
| Catalogue model per population, constructor overrides, graph `dt` | executed through the model contract |
| Signed weights, `random` / `all_to_all` rules, per-edge seeds, whole-step delays, autapse control | executed |
| Constant and Poisson drives per population | executed as public stimuli |
| Multiple projections between one pair | executed, currents add |
| Rust network runner | rejected (default parameters, no stimuli) |
| Per-synapse delay arrays, plasticity, state traces | not exposed on the canvas |
| Property editor for a projection's executed fields | weight, rule, probability, delay, seed, autapses; parsed in the browser, admitted by the server |
| Property editor for a population's model, count, type, drive and params | not yet; the fields are executed and still carry their creation defaults |
| Keyboard and screen-reader table equivalent of the canvas | rendered from the same graph, deletion included |
| Compiled (hardware) execution of a graph | separate unit; the pipeline compiles a fixed equation |
