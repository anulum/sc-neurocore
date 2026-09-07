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

## Duplicating part of a graph

Select one or more populations and press **Duplicate**. The copies carry every
field the runtime executes — model, count, type, drive and parameter overrides
— offset so they are visible as copies, and labelled distinguishably (`Exc 0
copy`, then `Exc 0 copy 2`).

**A projection is copied only when both of its ends are inside the
selection.** One that crosses the boundary is left behind, and the canvas says
how many were. There is no answer that is right for everyone: pointing the copy
at the original preserves a fan-in someone may have been reproducing, pointing
it at the copy preserves the motif's shape, and either is silently wrong for
the other reader. A wrong graph that ran is worse than one that refused.

Identifiers come from the server, through the same routes that mint them for
anything else; a duplicate never invents one. The whole operation is a single
undo step, not one per object created, and the copies are left selected so a
second duplicate copies the copy.

The projection creation route carries neither `seed` nor `autapses`, so a
duplicate applies those two from the original after the copy exists — without
that, a copy would silently run with a different connectivity draw.

## Editing a population

Click a population to open its property editor: label, model, neuron count,
neuron type, external input, and the model's own constructor parameters.

The parameters are the half that cannot be guessed. Which constructor fields a
population may override, each field's kind and its declared default are decided
by the run contract, and `GET /api/graph/models/{name}` states them:

```json
{
  "drive": {"kind": "float", "parameter": "current", "positional_only": false},
  "model": "SCLapicqueLIFNeuron",
  "parameters": [{"default": 1.1, "kind": "float", "name": "capacitance"}],
  "schema_version": "studio.population-model-contract.v1",
  "unsupported": [{"name": "dt", "reason": "the timestep is set through the dt field, not a parameter override"}]
}
```

Nothing the contract offers is refused on use: `dt` is overridable on the class
and refused as an override by the run contract, so it is reported under
`unsupported` — with that contract's own wording — rather than offered as an
input that is rejected every time. The fields that are not inputs are shown in
the editor with the reason each is not, because a user who cannot find one
should read why rather than conclude the editor is incomplete.

Until the contract arrives the editor offers **no** parameters and says so;
changing the model clears the previous model's overrides, which mean nothing to
the new one and which the graph would refuse. A model this canvas cannot
execute answers `404` and is not offered in the model list at all.

The external input follows the specification's shapes: a drive of kind `none`
carries no fields, `constant` carries a current, `poisson` carries a rate, an
event weight and an optional seed. Changing the kind replaces the drive with
one carrying only that kind's fields, because a drive carrying a foreign field
is refused.

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

## The browser contract

Component tests assert the markup a component renders. They cannot assert what
a browser *computes* from it: the accessible name of a control after labels,
`aria-label` and text content have been reconciled, whether a control is
reachable by keyboard, whether a hidden region leaves the tab order. Those are
the properties an assistive technology consumes, and only a browser produces
them.

```bash
cd studio/frontend && npm run test:e2e:graph
```

That builds the bundle and runs `e2e/network-canvas-live.spec.ts` against the
**built** bundle served by `vite preview`, proxied to a **real** Studio backend
— no mocked routes. It checks that the canvas builds a graph the live server
resolves and runs, that every visible control has a computed accessible name,
that the table is found by its caption and carries one row header per
population and a delete control naming what it removes, that the canvas is
hidden — and so leaves the tab order — while the table stands in for it, that
each editor input is bound to its label and to the statement of its contract,
that a refused value carries the server's own message to the input that caused
it, and that every editor input can take focus.

It also checks that a reader who has asked their system for less motion gets
none. Three progress indicators animate their width from an inline `style`
attribute, which a stylesheet reaches only through `!important`, so the
`prefers-reduced-motion` rule in `index.css` is written against `*`. The check
measures what the browser computes from the shipped stylesheet against an
inline declaration under both media states — 0.3 s without the preference, and
effectively nothing with it. Durations become `0.01ms` rather than `none` so
that `transitionend` handlers still fire and nothing waits for an event that
was cancelled.

### Colour contrast

Contrast is the one property here that has to be **computed** rather than
queried. A background read from the element itself is `rgba(0,0,0,0)` almost
everywhere, so the colour a reader actually sees is composited from the
ancestor chain, and an element whose background cannot be resolved — a
gradient, an image, a chain that never reaches an opaque colour — is
**reported, never scored**. Assuming white there would manufacture a pass for
dark text on an unknown ground. The thresholds are WCAG 2.2 AA: 4.5:1, or 3:1
for large text (18.66px bold, 24px otherwise).

When this audit was first run it found **ten failing colour pairs**, the worst
at **1.45:1** where 4.5:1 is required. Seven of them were a single decision:
`--text-muted` against seven different grounds. All ten are now fixed, so
`contrast-baseline.json` is empty; it stays in the tree because it works in
both directions — a pair it does not record fails the run, and a recorded pair
that no longer fails also fails the run — so a new failure has somewhere to be
refused and an old one cannot be quietly re-accepted.

The baseline is keyed on the **colours** rather than on the element or its
text. That distinction is not cosmetic: one palette decision fails on dozens of
unrelated elements, and the text on screen depends on which other tests ran
first, so a text-keyed list reported seventeen entries alone and nine hundred
and sixteen in a full suite. Keyed on colour it was ten either way.

### What the browser audit cannot see, and what covers it

A DOM audit can only read text nodes. Two surfaces carry ordinary product text
and are invisible to it: text painted into a `<canvas>`, which is pixels by the
time the DOM reports anything, and text written into an exported SVG, which
never reaches a page at all. Both were failing — every axis and tick label in
every Studio plot sat at about **2.3:1** — and neither would ever have appeared
in the browser run.

`src/paletteContrast.test.ts` covers them by reading the sources that declare
the colours: the stylesheet's own custom properties, the badge colour maps, the
plot constants, and a scan of every colour `SimulationPlot` assigns to
`fillStyle` immediately before painting text. It composites each translucent
ground over the base the component actually places it on, and it uses
`contrastAudit.ts`'s arithmetic rather than a second copy of it. A pair it
cannot resolve throws instead of passing, and a scan that stops matching the
source fails instead of reporting the file clean.

### How the grey ramp is chosen

`--text-muted` is the lowest luminance of its own blue-grey that still reaches
4.5:1 against **every** ground it is rendered on — the plain `--bg-*` surfaces,
`--accent-dim`, and the composited tints of the Network Canvas. The binding one
is the accent tint behind the **+ Exc** button, where it clears by 4.55:1.
`--text-secondary` then sits at the luminance midpoint between it and
`--text-primary`, so the ramp keeps three distinguishable steps in the order it
always had. Both values are consequences of the threshold, not preferences, and
changing either fails the test unless the new one also passes.

The same rule fixed the rest: the inhibitory `+ Inh` button's tint dropped from
0.2 to 0.12 alpha, because `#ff5252` is far darker than `#4fc3f7` and equal
alphas do not give equal contrast; `PLOT_AXIS` rose to `#727d8b`, which clears
the 4.5:1 text bar and therefore the 3:1 bar its axis rules need; and the two
badge greys rose to the least they could and still pass in both of the roles a
badge colour has — chip text while the filter is off, chip ground while it is
on.

### Non-text contrast, and where it stops

SC 1.4.11 asks for 3:1 from two things: the visual information that identifies
a **user interface component**, and the parts of a **graphic required to
understand the content**. Not from everything that has an edge.

A control's outline is the first of those — it is what says the thing is a
control — and `--border` at 1.42:1 was not enough for it. `--control-border`
(`#728091`) is the lowest luminance of the same blue-grey that clears 3:1 on
every surface a control sits on; `--bg-hover` binds it at 3.03:1. Every button,
input, select and textarea that used to draw itself with `--border` now uses
it, in the stylesheet and in the components.

`--border` stays where it was, at 1.42:1, and that is a decision rather than an
omission: panel edges, row separators and card outlines carry no information a
reader needs, so the criterion does not reach them. Grid lines behind the plot
data are the same case — redundant guides drawn beneath labelled axes, where
raising them would compete with the traces. Both figures are recorded in the
test so a later reader can see they were measured and excluded.

The split is worth guarding in two directions. The token check alone would stay
green if a control were drawn with `--border`, because both tokens pass their
own thresholds, so `controlsUsingDividerBorder` reads the component sources and
attributes each inline `border` declaration to the nearest opening tag; a
control found using the divider token fails the suite. A one-sided
`borderTop`/`borderBottom` is a divider by construction and is not considered.

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

Below the table, **Source** and **Target** choose two populations by name and
**Connect** creates the projection between them. On the canvas a projection is
made by dragging between node handles, which no keyboard reaches; without this
a network could be added to, edited, deleted from and undone without a mouse,
but never connected — so it could not be built at all. Both controls choose by
name rather than by position, and the button calls the same store action the
drag calls, including its Dale's-principle sign derivation and its failure
reporting, so the two paths cannot diverge. It says what it needs while only
one population exists, and stays disabled until both ends are chosen.

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
| GET | `/api/graph/models/{name}` | What a population of one model may override, and why the other fields are not inputs |
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
| Duplicate a selection, with the projections inside it | copies every executed field; a projection crossing the selection is reported, not guessed |
| Property editor for a projection's executed fields | weight, rule, probability, delay, seed, autapses; parsed in the browser, admitted by the server |
| Property editor for a population's model, count, type, drive and params | driven by `GET /api/graph/models/{name}`; parsed in the browser, admitted by the server |
| Keyboard and screen-reader table equivalent of the canvas | rendered from the same graph, deletion included |
| Compiled (hardware) execution of a graph | separate unit; the pipeline compiles a fixed equation |
