# Compiler Inspector

The Compiler Inspector lowers the equations in the ODE playground through the
native equation compiler. It shows the resulting SC intermediate representation
and SystemVerilog, the actual output widths and clock cadence, and separate
results for graph structure and source replay.

## Compile the current equations

1. Select **ODE mode** and enter the equations, initial values, parameters,
   threshold and reset rules in the playground.
2. Open **Equation compiler settings** in the Inspector. Choose the integration
   method, event rules, arithmetic format and optional physical units. The
   playground supplies the timestep.
3. Select **Build IR**. Inspect the actual operation and port counts, output
   widths, product registers and clocks per timestep.
4. Select **Check graph structure** to inspect the native graph, or **Replay
   source** to reconstruct its bound equations and require the same graph.
5. Select **Emit SV** to generate native SystemVerilog. This builds the current
   equations first when no matching graph exists. Emission replays bound source
   before generating RTL.
6. Download the IR, SystemVerilog and compiler request. Export the compile
   evidence bundle to retain the worker artifacts and source replay command.

Changing any compiler input withdraws the previous IR, RTL and validation
results. A response from an older request cannot restore them. Starting another
compiler action while work is running does not submit a duplicate job.

## Equations, events and arithmetic

The native request retains the authored equation order and nonzero initial
state. The integration choices are Euler, RK4, exponential Euler, discrete map
and Gauss–Seidel. For Gauss–Seidel, changing the equation order can change the
model; the order survives worker transport and source replay.

The settings include:

- Level, crossing, Poisson and escape-rate detection; reset expressions and
  optional graded spike values.
- An exact unsigned 64-bit compiler random key and unsigned 32-bit stream. The
  key is decimal text throughout the browser and request. It is separate from
  the simulation seed.
- Explicit immutable constants selected from the playground parameters.
- Total scalar bits, fractional bits, signedness, overflow and rounding.
- Global product pipeline stages or named product pipeline points.
- The default `lut-v1` function profile and opt-in `v2` profile; normal-draw
  precision is selected explicitly.
- Strict physical units for initial states, parameters, constants, timestep
  and the input drive. Unit declarations preserve the entered display units.

The compiler determines which combinations it supports. An unsupported
combination is refused; the caller does not replace it with Q8.8, another
integrator or another event rule. For example, multiple substeps currently
require crossing detection without reset rules. Unsigned equation arithmetic
is not an advertised supported datapath.

Compiler settings travel with saved projects and local sessions. Incomplete
text drafts remain editable after reload, but cannot compile until valid.
Projects without the compiler settings block receive explicit legacy defaults;
an explicitly malformed or unknown settings block is refused.

## What the checks establish

| Result | Meaning |
|---|---|
| Graph structure valid | The native parser and verifier accept the graph's operations, references, types and ports. |
| Source replay not run | Bound equation source is present, but this action did not reconstruct it. |
| Source replay passed | Recompiling the bound model and profile produced the submitted graph. |
| Source replay unavailable | The graph has no bound equation source. |
| SystemVerilog emitted | The native emitter returned RTL after the required structural and source checks. |

Structural validity does not establish source consistency. Source replay does
not establish scientific accuracy, successful RTL tool compilation, synthesis
or physical hardware behaviour. Use the [hardware user guide](manual/hardware.md)
for the subsequent qualification steps.

The native equation datapath uses explicit scalar state, integer word
operations, stores and output ports. Its IR is not a fixed LIF template.
The emitter supplies its own clock and reset ports; they are not authored
input declarations.

## HTTP API

All native build, inspection, replay and emission actions use the bounded
Studio process service. Responses include a path-free `studio_job_receipt`
with the actual job identifier and artifact digests. A completed worker can
carry a refused compilation; worker completion and successful compilation
are separate outcomes.

| Endpoint | Request and result |
|---|---|
| `POST /api/ir/build` | Versioned authored model and profile; native graph and output contract. |
| `POST /api/ir/verify` | `ir_text`; structural inspection without a source replay claim. |
| `POST /api/ir/replay` | `ir_text`; require bound source and reconstruct the same graph. |
| `POST /api/ir/emit-sv` | `ir_text`; native SystemVerilog, requiring source replay by default. |
| `POST /api/ir/emit-sv-direct` | Separate direct Python equation compiler and its traceability manifest. |

A minimal native build request is:

```json
{
  "schema_version": "studio.equation-ir-request.v1",
  "model": {
    "equations": [{"name": "v", "expression": "I"}],
    "initial_state": [{"name": "v", "value": 7}],
    "dt": {"value": 0.5},
    "rng_seed": "18446744073709551615"
  },
  "profile": {
    "module_name": "authored_neuron",
    "data_width": 16,
    "fraction": 8
  }
}
```

Ordered declarations are arrays in the API. The request schema rejects repeated
names, unknown fields, lossy random keys and the former unversioned simulation
bodies. Invalid or unsupported compiler inputs return a controlled refusal;
unavailable compiler dependencies are distinguished from invalid source.
Malformed worker results return status 500 with `studio_compiler_result_invalid`;
process completion alone never makes a malformed compiler result acceptable.

Raw native graphs without bound equation source cannot pass source replay.
An API caller can explicitly request `source_policy: "structural-only"` for
emission of an unsourced graph. Its response still reports source replay as
unavailable. A graph with source metadata must replay successfully even when
structural-only emission is requested.

## The direct Python compiler

**Generate RTL** uses the separate direct Python equation compiler. Its
`studio.compile-traceability.v1` manifest binds the submitted equations,
initial state, parameters, threshold and reset to the generated RTL. The
Inspector retains its source, RTL and traceability digests and evidence export.
The native **Emit SV** action does not silently switch to this route.

Neither compilation path alone qualifies a model for a hardware target.
Retain the compiler/profile identity and run the applicable numerical,
co-simulation and tool checks for the emitted design.
