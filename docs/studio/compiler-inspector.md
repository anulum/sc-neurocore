# Compiler Inspector

The Compiler Inspector lets you see how your neuron ODE equations compile
to stochastic computing hardware. It shows the intermediate representation
(IR) and generated SystemVerilog side by side, with verification status
and Q8.8 fixed-point parameter encoding.

The inspector keeps the generated RTL and its source traceability visible
for review before downstream synthesis or hardware validation.

## Current equation-translation boundary

The Studio Rust `build_ir_from_equation` adapter currently constructs a fixed
LIF graph. Submitted equations, threshold and reset are not lowered into their
corresponding operations. A structurally valid IR graph therefore does not
establish that the entered equations were compiled. Treat **Build IR** as
inspection of that template, not general equation-to-hardware evidence.

The direct Python equation compiler is a separate route with its own
supported-operation contract and source-to-RTL traceability. Generated source
still needs numerical, RTL and tool qualification. For a task-oriented path,
read the [hardware user guide](manual/hardware.md).

## Inspection workflow

1. Switch to **ODE mode** and retain the equations and numerical inputs.
2. Use **Build IR** only with the fixed-template limitation above in mind.
3. Inspect its IR text, emitted source and structural-verification diagnostics.
4. Use **Generate RTL** / the direct Python path when evaluating supported
   equation compilation; **Emit SystemVerilog** also invokes the direct path.
5. Inspect source/RTL digests, module name and compile evidence class.
6. Qualify emitted source through the appropriate simulation and tool workflow;
   neither a green structural badge nor source emission proves synthesisability.

## What the IR Represents

The SC IR (Stochastic Computing Intermediate Representation) is a dataflow
graph where each node is a hardware operation:

| Operation | Hardware Meaning |
|-----------|-----------------|
| `input` | Port (clock, current) |
| `constant` | Fixed-point literal |
| `encode` | Bernoulli bitstream encoder (LFSR + comparator) |
| `bitwise_and` | Stochastic multiplication (P(A)·P(B)) |
| `popcount` | Bitstream → integer (population count) |
| `lif_step` | LIF neuron pipeline (leak, gain, threshold, reset) |
| `scale` | Multiply by constant factor |
| `offset` | Add constant |
| `output` | Output port (spike signal) |

The IR uses SSA (static single assignment) form. Each value has exactly
one definition and can be used by multiple consumers.

## Verification

The IR verifier checks:

- **DAG structure**: no cycles in the dataflow graph
- **SSA validity**: every operand references a defined value
- **Type consistency**: operations receive compatible types
- **Port completeness**: all inputs connected, all outputs driven

Verification runs automatically when you build the IR. Errors appear
in the red banner above the split pane.

## Q8.8 Fixed-Point Encoding

The current Rust template uses Q8.8 constants. In this format:

- 8 integer bits + 8 fractional bits = 16-bit signed
- Range: -128.0 to +127.996
- Resolution: 1/256 ≈ 0.0039

The parameter table shows each parameter's float value alongside its
Q8.8 encoded integer. This is the exact value used in the generated
SystemVerilog module.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/ir/build` | Equation → ScGraph → IR text |
| POST | `/api/ir/verify` | Parse and verify IR text |
| POST | `/api/ir/emit-sv` | IR text → SystemVerilog |
| POST | `/api/ir/emit-sv-direct` | Equation → Verilog plus compile traceability |
| POST | `/api/ir/cosim` | Float vs Q8.8 co-simulation |

## Two Compilation Paths

1. **Rust IR inspector** (**Build IR**): fixed LIF template → ScGraphBuilder
   → IR → SystemVerilog. Uses structural verification; it does not currently
   translate arbitrary submitted equations into equivalent hardware operations.

2. **Python compiler path** (`SV` button): Equation → AST → Q8.8 Verilog.
   Uses `equation_compiler.py` to generate Verilog directly from the equation
   AST. Produces register-transfer-level Verilog with explicit fixed-point
   arithmetic.

Source emission is a separate result from successful tool compilation,
co-simulation and synthesis. Do not infer equation fidelity from template
verification or physical device support from either path.

## Source-to-RTL Traceability

The direct equation-to-Verilog path returns a path-free
`studio.compile-traceability.v1` manifest in `compile_traceability`.
The manifest records:

- the submitted equations, threshold, reset, initial state, and numeric
  parameter map;
- `input_sha256`, a stable SHA-256 digest over the canonical source payload;
- the emitted RTL language, module name, source length, and `rtl_sha256`;
- `traceability_sha256`, a stable digest over the complete public manifest;
- `evidence_classification: "compile"` and `status: "completed"` for
  evidence-bundle consumers.

The traceability serializer validates its evidence class and terminal status
through the shared Studio evidence-classification contract before returning
public metadata.

The manifest is included in both `/api/compile` and
`/api/ir/emit-sv-direct` responses. The Compiler Inspector displays the
schema version, evidence class, terminal status, module name, and shortened
source/RTL/manifest digests beside the generated RTL. Operators can export a compile evidence
bundle directly from that traceability strip; the bundle records replay
metadata for `/api/ir/emit-sv-direct`, the compile input digest, and the
traceability digest note through the same admin evidence-bundle route used by
the Admin panel. It deliberately contains no host-local paths.

Adaptive-precision formal bundles created through
`/api/adaptive-precision/formal-bundle` use the same evidence-classification
contract for their generated SymbiYosys-ready manifests:
`evidence_classification: "compile"` and `status: "completed"` mean the bundle
generation completed, while the embedded formal-claim flags remain fail-closed
until external SymbiYosys execution evidence is attached.

The adaptive-precision planner accepts finite vector or matrix layer weights
and rejects empty, non-finite, or higher-rank inputs before stochastic
sensitivity sampling. This keeps Studio auto-tune and formal-bundle generation
deterministic across NumPy runtime and coverage-instrumented environments.

## Co-Simulation

The **Q8.8 precision** view and `/api/ir/cosim` compare a float64 run with
execution of the generated bit-true fixed-point kernel under the selected
numeric contract. This is separate from RTL-versus-C catalogue co-simulation.
It reports:

- **max_error**: worst-case absolute difference
- **mean_error**: average difference
- **rms_error**: root mean square difference
- **Error trace**: per-timestep absolute error

This quantifies the precision loss from float-to-hardware conversion.
