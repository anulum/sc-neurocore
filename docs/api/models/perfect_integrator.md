# PerfectIntegratorNeuron

**Module:** `sc_neurocore.neurons.models.perfect_integrator`
**Reference:** Lapicque 1907 (no-leak variant)
**Family:** Integrate-and-fire (non-leaky)
**State variables:** `v` (voltage)

## Equations

$$C_m \frac{dV}{dt} = I$$

Discrete: $V(t+1) = V(t) + \frac{I}{C_m} \cdot dt$

Spike when $V \geq V_\theta$, then $V \leftarrow V_{\text{reset}}$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `v` | 0.0 | Membrane voltage |
| `c_m` | 1.0 | Membrane capacitance |
| `v_threshold` | 1.0 | Spike threshold |
| `v_reset` | 0.0 | Reset potential |
| `dt` | 0.1 | Time step |

## Validation contract

The implementation rejects invalid state before mutation:

- `v`, `v_threshold`, `v_reset`, `c_m`, `dt`, and input current must be finite;
- `c_m` and `dt` must be positive;
- `v_threshold` must be greater than `v_reset`;
- initial `v` must be below `v_threshold`;
- each voltage increment and candidate voltage must remain finite before assignment.
- runtime `v`, `c_m`, `dt`, `v_threshold`, and `v_reset` are revalidated before
  the `I / C_m` division so corrupted objects fail closed without mutating
  voltage.

These guards preserve the analytical positive-excursion ISI contract and
prevent overflowing currents or capacitance scales from poisoning the state.
Julia, Go, and Mojo transport the complete numeric contract through executable
native paths. Go and Mojo validate a complete run before writing the caller's
trace, so rejection cannot leave partial output. The Rust engine path is
executable at factory defaults and rejects non-default instances explicitly.

The schema-level reference corpus pins the spike-bearing constant-current
protocol `perfect_integrator_constant_current_sawtooth`. Its features are
re-derived independently from the analytic reset sawtooth in
`tests/test_reference_perfect_integrator.py`.

## Behaviour

- **No leak:** zero-input steps leave voltage unchanged; unlike LIF, there is no
  drift toward a resting potential.
- **Candidate-first threshold:** the Euler candidate is computed and checked
  before an inclusive threshold comparison and hard reset.
- **Linear f–I relation:** below the one-event-per-step ceiling, firing rate is
  proportional to current and inversely proportional to capacitance and the
  threshold excursion.
- **Deterministic:** identical state and inputs produce bit-identical traces.
- **Floating-point boundary:** decimal increments need not reach a decimal
  threshold on the algebraically expected step. The tests retain this IEEE 754
  behaviour instead of replacing it with an epsilon threshold.

## Analytical predictions

| Property | Formula |
|----------|---------|
| ISI (steps) | $\lceil (\theta - V_{\text{reset}}) / (I \cdot dt / C_m) \rceil$ |
| Rate | $I / (C_m \cdot (\theta - V_{\text{reset}}))$ before the discrete ceiling |
| Linearity | $f(2I) = 2 f(I)$ away from quantisation boundaries |
| Capacitance scaling | $f \propto 1/C_m$ |
| Threshold scaling | $f \propto 1/(\theta - V_{\text{reset}})$ |

## Execution and silicon pipeline

```
PerfectIntegratorNeuron
├── step(current) → int {0,1}
├── simulate(..., backend="auto|python|rust|julia|go|mojo")
├── measured auto order: Mojo → Julia → Go → compatible Rust → Python
├── paired TOML/JSON schema: Euler + inclusive candidate threshold
├── generated Q8.8 RTL: 66-event parity at I=0.7 over 1,000 steps
└── catalogue formal job: SymbiYosys/Z3 bounded proof, depth 20
```

## Verification evidence

| Surface | Evidence | Contract |
|---------|----------|----------|
| Python model | `tests/test_model_perfect_integrator.py` | dynamics, f–I/ISI laws, reset, validation, analysis, and network use |
| Public native dispatch | `tests/test_perfect_integrator_backend_parity.py`, `tests/test_perfect_integrator_backend_auto_dispatch.py`, `tests/test_perfect_integrator_backend_validation.py`, `tests/test_perfect_integrator_backend_c_abi.py`, `tests/test_perfect_integrator_backend_unavailability.py` | executable Rust/Julia/Go/Mojo paths, bit-exact parity, full numeric contract, and mutation-free rejection |
| Native loading | `tests/test_perfect_integrator_backend_loading.py` | fail-closed optional-runtime and C-symbol boundaries |
| Analytic reference | `tests/test_reference_perfect_integrator.py` | independent reset-sawtooth feature re-derivation |
| Python-to-Verilog | `tests/test_cosim_perfect_integrator.py` | hand/schema/Q8.8 parity plus an explicit fractional-current boundary |
| Benchmark | `tests/test_bench_perfect_integrator.py` | public-path measurement, source hashes, environment metadata, and fail-closed parity exits |

The acceleration goldens cover 1,000 steps at I=0/0.333/0.7/2/3/5/20,
producing 0/32/66/200/250/500/1,000 events. Every Rust, Julia, Go, and Mojo
trace is bit-identical to Python. At I=0.7, hand Python, schema Python, and
Q8.8 RTL all produce 66 events over 1,000 steps. At I=0.333, fixed-point
quantisation produces 31 RTL events versus 32 in both floating-point paths;
that one-event boundary is a declared exclusion, not a failed parity claim.

## Measured performance (2026-07-13)

The committed run was pinned to logical CPU 10, but that CPU was not reserved
and the kernel isolated-CPU set was empty. The powersave-governor host load was
30.28 at the start and 30.76 at the end. These are local regression timings,
not production throughput claims.

| Metric | Value |
|--------|-------|
| Evidence class | Local regression, non-isolated workstation |
| Benchmark artefact | `benchmarks/results/local_python_2026-07-13_perfect_integrator_euler.json` |
| Workload | 100,000 steps, 7 repeats, I=5.0 |
| Polyglot contract | Five public dispatch paths; 50,000 events and bit-exact voltage traces in every lane |

| Backend | Median ms/call | Speedup vs Python | Maximum voltage difference | Events |
|---------|---------------:|------------------:|---------------------------:|-------:|
| Mojo | 0.965 | 152.42× | `0` | 50,000 |
| Julia | 1.633 | 90.10× | `0` | 50,000 |
| Go | 2.644 | 55.64× | `0` | 50,000 |
| Rust engine | 60.268 | 2.44× | `0` | 50,000 |
| Python | 147.137 | 1.00× | `0` | 50,000 |

## Pipeline verification

1. Construction, scalar stepping, reset, population use, and long-run state
   stability pass through the maintained Python model suite.
2. Rust, Julia, Go, and Mojo execute the same candidate-first recurrence. Julia,
   Go, and Mojo carry non-default state and parameters; Rust retains its stated
   factory-default boundary.
3. Hand/schema/Q8.8 RTL preserve the enrolled 66-event operating point, and the
   I=0.333 quantisation boundary remains explicit.
4. The generated inclusive-threshold RTL passes the depth-20 SymbiYosys/Z3
   bounded proof.
