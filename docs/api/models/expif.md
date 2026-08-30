# ExpIFNeuron

`ExpIFNeuron` implements the exponential integrate-and-fire current balance of
Fourcaud-Trocmé, Hansel, van Vreeswijk and Brunel (2003). It deliberately
exposes two profiles instead of rewriting a long-lived SC numerical contract:

- `ExpIFNeuron.fourcaud_trocme_2003()` is the source-aligned deterministic profile;
- `ExpIFNeuron()` and `ExpIFNeuron.sc_rk4_compatibility()` preserve the
  historical SC candidate-first RK4/Q32.32 profile.

Primary source: N. Fourcaud-Trocmé et al., *Journal of Neuroscience* 23(37),
11628–11640 (2003),
[doi:10.1523/JNEUROSCI.23-37-11628.2003](https://doi.org/10.1523/JNEUROSCI.23-37-11628.2003).

## Equation and current convention

The source defines

\[
C\frac{dV}{dt}=-g_L(V-V_L)
  +g_L\Delta_T\exp\!\left(\frac{V-V_T}{\Delta_T}\right)+I_{syn}(t).
\]

After division by `gL`, the runtime evaluates

\[
\tau\frac{dV}{dt}=-(V-V_{rest})
  +\Delta_T\exp\!\left(\frac{V-V_{rh}}{\Delta_T}\right)+I,
\quad \tau=C/g_L.
\]

The public `current` is therefore voltage-equivalent `Isyn/gL`, not an
unscaled current in picoamperes.

## Source profile

```python
from sc_neurocore.neurons.models.expif import ExpIFNeuron

neuron = ExpIFNeuron.fourcaud_trocme_2003()
voltage, refractory, events = neuron.simulate_complete(
    4_000, current=20.0, backend="rust"
)
```

The factory freezes the fitted values and source simulation boundary:

| Quantity | Value | Source role |
|---|---:|---|
| `C` | 1 µF/cm² | capacitance density |
| `gL` | 0.1 mS/cm² | leak conductance density |
| `v_rest` | −65.0 mV | leak reversal |
| `v_rh` | −59.9 mV | soft exponential threshold |
| `delta_t` | 3.48 mV | exponential slope factor |
| `v_reset` | −68.0 mV | reset potential |
| `tau` | 10.0 ms | `C/gL` |
| `v_threshold` | −30.0 mV | numerical handoff to the analytical tail |
| `refractory_period` | 1.7 ms | fitted refractory duration |

Below −30 mV the paper uses stochastic second-order Runge–Kutta with a
timestep less than 0.02 ms. Above −30 mV it neglects all but the exponential
current and analytically integrates the remaining finite time to divergence.
The separate +20 mV value in the paper is an observation threshold used to
define spike times in comparisons; it is not the numerical handoff.

The maintained source-aligned lane is the deterministic zero-noise Heun specialization
with `dt=0.01 ms`. The source specifies only `dt < 0.02 ms`, so 0.01 ms is an
explicit numerical choice backed by 0.01/0.005 ms convergence evidence. The
analytical exponential-only tail from −30 mV is
`0.001855930799631619 ms`.

## SC compatibility profile

The zero-argument constructor retains the previously published SC behaviour:

| Field | SC default |
|---|---:|
| `v_threshold` | +30.0 mV |
| `dt` | 0.02 ms |
| integrator | candidate-first, stage-clipped classical RK4 |
| `refractory_period` | 0.0 ms |

This recurrence remains useful and is implemented across Python, Rust, Julia,
Go, Mojo, paired executable schemas and Q32.32 RTL. It is not described as the
paper's simulation protocol. The generated RTL, co-simulation and formal
proofs apply to this compatibility lane only.

## Complete batch contract

`simulate_complete()` returns aligned post-step arrays:

```text
(voltage: float64[n], refractory_remaining: float64[n], events: uint8[n])
```

Every backend accepts the complete state and parameter surface. Python,
production Rust/PyO3, safety Rust, Julia, Go and Mojo execute against a
candidate state and commit the receiver only after the whole packet succeeds.
Malformed shapes, non-finite states and non-binary events are rejected before
commit. The source-profile selector additionally fails closed unless the fitted
source values and strict `dt < 0.02 ms` bound are intact; arbitrary parameter
studies remain available under the SC profile. `simulate()` remains the
compatibility facade returning voltage and an aggregate spike count.

`auto` preserves the measured Julia → Go → Mojo → Rust → Python order. An
explicit selector is one of `python`, `rust`, `julia`, `go`, or `mojo`.

## Production routing and Studio

The compiled NetworkRunner accepts all established names:

- `ExpIFNeuron` constructs the source profile;
- `ExpIF` and mixed-case `ExpIfNeuron` preserve the SC compatibility profile.

The installed PyO3 function `expif_simulate_complete` crosses into Rust once
per batch. Studio consumes `ExpIFNeuron.toml` through its real catalogue and
HTTP detail endpoints; it exposes the source receipt, parameters, backend
facets, validation evidence and compile configuration.

## Reproducibility evidence

The independent source receipt is
`src/sc_neurocore/neurons/reference_receipts/expif_fourcaud_trocme_2003.json`.
Its 20,000-step five-segment input schedule binds the complete voltage,
refractory, event and current arrays with SHA-256 digests, exact event indices,
final state, primary-source fit, inspected-paper digest and analytical-tail
derivation. `tests/test_reference_expif_source_receipt.py` re-derives that
packet without importing the production model.

Focused evidence anchors:

- `tests/test_model_expif_source_contract.py` — paired schema/profile identity;
- `tests/test_expif_backends_batch_atomicity.py` — five-runtime complete packets,
  source parity, convergence and atomicity;
- `tests/test_expif_engine_binding.py` — installed PyO3 and NetworkRunner E2E;
- `tests/test_reference_expif_source_receipt.py` — independent full receipt;
- `tests/test_cosim_exp_if.py` — SC schema and Q32.32 RTL co-simulation;
- `hdl/formal/catalogue/sc_exponential_if.sby` — bounded reset/event safety;
- `hdl/reports/yosys_expif_q3232_2026-08-30.json` — generic synthesis receipt.

The tracked compatibility core synthesizes to 484,938 coarse cells in Yosys
0.33. This is an executable H2 compile receipt, not a timing, PPA, or target
device estimate.

The benchmark producer is `benchmarks/bench_model_expif.py`; the committed
result is `benchmarks/results/bench_expif.json`. It is a source/binary-bound
local regression record, not a universal throughput, timing, PPA, device or
physical-silicon claim.

## Scope limits

- The source profile is a deterministic zero-noise specialization of a paper
  protocol used with fluctuating stochastic input.
- Macro-step event rows are discrete; the receipt separately preserves the
  analytical tail duration required to refine a continuous source spike time.
- Continuous spike timestamps and stochastic-input realisations are not part
  of the current complete-packet contract.
- Q32.32 RTL evidence belongs to the SC RK4 profile, not the source RK2 lane.
- Timing closure, PPA, named FPGA/ASIC mapping, board/HIL and physical silicon
  remain outside the current evidence boundary.
