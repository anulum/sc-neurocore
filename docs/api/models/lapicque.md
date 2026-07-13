# LapicqueNeuron

**Module:** `sc_neurocore.neurons.models.lapicque`

**Reference:** Lapicque 1907; English translation DOI `10.1007/s00422-007-0189-6`

**Family:** Integrate-and-fire (classical)

**State variables:** `v` (voltage)

## Equations

$$\tau \frac{dV}{dt} = -(V - V_r) + R \cdot I$$

Spike: $V \geq V_\theta$, hard reset $V \to V_{reset}$.

For constant current over a timestep, maintained runtime surfaces use the exact
RC flow rather than forward Euler:

$$V_{t+\Delta t} = V_\infty + (V_t - V_\infty)e^{-\Delta t/\tau}$$

where $V_\infty = V_r + R \cdot I$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau` | 20.0 | Membrane time constant (ms) |
| `resistance` | 1.0 | Membrane resistance |
| `v_threshold` | 1.0 | Spike threshold |
| `v_reset` | 0.0 | Post-spike reset |
| `dt` | 1.0 | Integration step |

## Validation contract

The implementation rejects invalid state before mutation:

- `v`, `v_rest`, `v_reset`, `v_threshold`, `tau`, `resistance`, `dt`, and input current must be finite;
- `tau`, `resistance`, and `dt` must be positive;
- `v_threshold` must be greater than both `v_rest` and `v_reset`;
- initial `v` must be below `v_threshold`;
- exact-flow steady voltage, decay, and candidate voltage must remain finite
  before assignment.

These guards preserve the positive-rheobase RC contract and prevent overflowing
inputs or time constants from poisoning membrane state.

Python re-validates mutable runtime state on every `step()` call. Julia, Go, and
Mojo expose the complete state-and-parameter contract through executable native
ABIs. Go and Mojo validate the complete run before writing the caller-visible
trace, so rejected input cannot partially commit output or instance state. The
Rust engine path is executable at factory defaults and rejects non-default
instances explicitly instead of silently changing their parameters.

## Behaviour

- **Historical RC formulation:** the maintained recurrence follows Lapicque's
  1907 polarisation model with a threshold and hard reset.
- **Analytical rheobase:** I_rh = V_θ / R. Below rheobase, v settles to
  steady state R·I < V_θ. Above, periodic spiking.
- **Deterministic:** Fully deterministic exact constant-current RC integration.
- **Hard reset:** v → v_reset (not subtract-reset).
- **Conductance-free point model:** no gating, adaptation, or noise state.

## Execution and silicon pipeline

```
LapicqueNeuron
├── step(current) → int {0,1}
├── simulate(..., backend="auto|python|rust|julia|go|mojo")
├── measured auto order: Mojo → Julia → Go → compatible Rust → Python
├── paired TOML/JSON schema: exp_euler + inclusive candidate threshold
├── generated Q16.16 RTL: event-vector parity at three operating points
└── catalogue formal job: SymbiYosys/Z3 bounded proof, depth 20
```

## Verification evidence

| Surface | Evidence | Contract |
|---------|----------|----------|
| Python model | `tests/test_model_lapicque.py` | exact flow, rheobase, reset, validation, analysis, network use, and timing guard |
| Public native dispatch | `tests/test_lapicque_backends.py` | executable Rust/Julia/Go/Mojo paths, complete parity, measured fall-through order, and mutation-free rejection |
| Native loading | `tests/test_lapicque_backend_loading.py` | build/load separation, ABI declarations, cache behaviour, and actionable failures |
| Reference | `tests/test_reference_lapicque.py` | independent closed-form feature re-derivation at `1e-12` absolute tolerance |
| Python-to-Verilog | `tests/test_cosim_lapicque.py` | paired-schema event exactness, `2e-15` state envelope, and Q16.16 event-vector parity |
| Benchmark | `tests/test_bench_lapicque.py` | public-path measurement, source hashes, environment metadata, partial-run disclosure, and fail-closed parity exits |

The model and native-dispatch modules reach 100 percent statement and branch
coverage under the focused closure cohort. The benchmark module reaches the
same configured threshold; its command-line entry point is also exercised by
the committed real measurement.


---

## Measured Performance (2026-07-13)

The committed run was pinned to logical CPU 10, but that CPU was not reserved
and the kernel isolated-CPU set was empty. The powersave-governor host load was
29.73 at the start and 30.47 at the end. These are local regression timings,
not production throughput claims.

| Metric | Value |
|--------|-------|
| Evidence class | Local regression, non-isolated workstation |
| Benchmark artefact | `benchmarks/results/local_python_2026-06-17_lapicque_exact_flow.json` |
| Workload | 100,000 steps, 7 repeats, I=5.0 |
| Polyglot contract | Five public dispatch paths; 20,000 events in every lane; maximum voltage difference `4.44e-16` |

| Backend | Median ms/call | Speedup vs Python | Maximum voltage difference | Events |
|---------|---------------:|------------------:|---------------------------:|-------:|
| Mojo | 1.075 | 256.07× | `4.44e-16` | 20,000 |
| Julia | 6.147 | 44.77× | `0` | 20,000 |
| Go | 8.138 | 33.82× | `0` | 20,000 |
| Rust engine | 70.889 | 3.88× | `0` | 20,000 |
| Python | 275.196 | 1.00× | `0` | 20,000 |

---

## Pipeline Verification (End-to-End)

### 1. Construction
`LapicqueNeuron()` instantiates with documented defaults.
**Status: PASS**

### 2. step() → correct type
Returns an integer spike indicator in `{0, 1}`.
**Status: PASS**

### 3. Spiking behaviour
2000 spikes in 10,000 steps at I=5.0.
**Status: PASS**

### 4. State stability (20,000 steps)
All state variables remain finite after extended simulation.
**Status: PASS**

### 5. reset()
State returns to initial values after `reset()`.
**Status: PASS**

### 6. Population
`Population(LapicqueNeuron, n=10)` creates correct instances.
**Status: PASS**

### 7. Public polyglot dispatch
Rust, Julia, Go, and Mojo execute the same exact-flow event contract. Julia,
Go, and Mojo carry non-default state and parameters; Rust retains its stated
factory-default boundary.
**Status: PASS**

### 8. Python-to-Verilog parity
Hand, TOML, and JSON traces agree to `2e-15`. Q16.16 RTL preserves the complete
event vectors at I=0.333/2.3/20.25 over 1,000 steps (0/83/500 events) with
maximum voltage error below `0.04`.
**Status: PASS**

### 9. Formal catalogue job
The generated exponential-Euler RTL and inclusive threshold contract pass the
depth-20 SymbiYosys/Z3 bounded proof.
**Status: PASS**

---

## Findings (measured 2026-07-13)

1. Constant-current integration uses the closed-form RC update on every public
   runtime path.
2. The measured public dispatcher order is Mojo, Julia, Go, compatible Rust,
   then Python.
3. All enrolled acceleration events match Python exactly; the largest measured
   trace difference is `4.44e-16`.
4. Paired schemas, Q16.16 RTL, readiness evidence, and the depth-20 formal job
   describe the same inclusive candidate-first event contract.
