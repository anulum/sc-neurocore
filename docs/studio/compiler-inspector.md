# Compiler Inspector

The Compiler Inspector lets you see how your neuron ODE equations compile
to stochastic computing hardware. It shows the intermediate representation
(IR) and generated SystemVerilog side by side, with verification status
and Q8.8 fixed-point parameter encoding.

No other SNN framework exposes this pipeline visually.

## Quick Start

1. Switch to **ODE mode** and enter your equations
2. Click **IR** to build the stochastic computing IR graph
3. The left pane shows the IR text; the right pane shows SystemVerilog
4. A green/red badge indicates verification status
5. Click **SV** for direct equation-to-Verilog via the Python compiler
6. Inspect the source-to-RTL traceability strip for the source digest,
   RTL digest, module name, and compile evidence class

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

All ODE parameters are converted to Q8.8 signed fixed-point format:

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

1. **Rust IR path** (`IR` button): Equation → ScGraphBuilder → IR → SystemVerilog.
   Uses the Rust IR compiler with SSA verification. Produces hardware-level
   stochastic computing operations (encode, bitwise_and, popcount, lif_step).

2. **Python compiler path** (`SV` button): Equation → AST → Q8.8 Verilog.
   Uses `equation_compiler.py` to generate Verilog directly from the equation
   AST. Produces register-transfer-level Verilog with explicit fixed-point
   arithmetic.

Both paths produce synthesisable output. The Rust IR path is more suitable
for stochastic computing architectures; the Python path targets conventional
fixed-point RTL.

## Source-to-RTL Traceability

The direct equation-to-Verilog path returns a path-free
`studio.compile-traceability.v1` manifest in `compile_traceability`.
The manifest records:

- the submitted equations, threshold, reset, initial state, and numeric
  parameter map;
- `input_sha256`, a stable SHA-256 digest over the canonical source payload;
- the emitted RTL language, module name, source length, and `rtl_sha256`;
- `traceability_sha256`, a stable digest over the complete public manifest;
- `evidence_classification: "compile"` for evidence-bundle consumers.

The manifest is included in both `/api/compile` and
`/api/ir/emit-sv-direct` responses. The Compiler Inspector displays the
schema version, evidence class, module name, and shortened source/RTL/manifest
digests beside the generated RTL. It deliberately contains no host-local paths.

## Co-Simulation

The co-simulation view (accessible via the existing Q8.8 tab or the
`/api/ir/cosim` endpoint) runs both a float64 Python simulation and a
Q8.8 fixed-point simulation side by side. It reports:

- **max_error**: worst-case absolute difference
- **mean_error**: average difference
- **rms_error**: root mean square difference
- **Error trace**: per-timestep absolute error

This quantifies the precision loss from float-to-hardware conversion.
