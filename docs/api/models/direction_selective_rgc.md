# DirectionSelectiveRGC

**Module:** `sc_neurocore.neurons.models.direction_selective_rgc`
**Rust path:** `sc_neurocore_engine::neurons::sensory::DirectionSelectiveRGC`
**Rust source:** `engine/src/neurons/sensory/direction_selective_rgc.rs`
**Reference:** Gollisch & Meister (2010), Masland (2012)
**Family:** Retinal sensory neuron
**State:** `v`, previous centre intensity, surround estimate

`DirectionSelectiveRGC` is a compact On/Off retinal ganglion-cell model for
motion-edge preprocessing. It computes a temporal centre derivative, subtracts a
low-pass surround estimate, and advances membrane voltage with exact
first-order relaxation for the drive held over one step.

## Equations

Temporal centre derivative:

$$\Delta I = I(t) - I(t-1)$$

Centre response:

$$R_c = \begin{cases} w_c \Delta I & \text{On-centre} \\ -w_c \Delta I & \text{Off-centre} \end{cases}$$

Surround estimate:

$$S(t) = 0.9 S(t-1) + 0.1 I_{surround}$$

Drive:

$$D = R_c - w_s S(t)$$

Exact membrane update:

$$V(t+dt) = D + (V(t)-D)\exp(-dt/\tau)$$

Spike/reset:

$$\text{spike}=1 \text{ if } V(t+dt) \ge \theta, \quad V \leftarrow 0$$

## Numerical contract

The maintained contract is candidate-first and fail-closed:

- centre and surround intensities must be finite and non-negative,
- `tau`, `theta`, and `dt` must remain finite and positive,
- centre/surround weights and runtime buffers must remain finite and non-negative,
- candidate surround, drive, decay, and voltage must be finite before commit,
- invalid runtime input or corrupted buffers preserve prior state,
- spike reset applies only to voltage; temporal buffers commit after a valid candidate.

The exact membrane update replaces the previous raw Euler increment. This avoids
time-step-dependent overshoot while preserving the temporal derivative and
centre-surround computation.

## Maintained implementation surfaces

| Surface | Contract |
|---------|----------|
| Python reference | Exact membrane relaxation with typed fail-closed errors |
| Rust engine | Exact membrane relaxation with no-spike sentinel on invalid runtime state |
| Go service | `StepRF(intensity, surround)` returns `(spike, error)` and preserves state on error |
| Julia mirror | `step_rf!` returns `-1` on invalid input and preserves state |
| Rust safety mirror | Standalone exact-relaxation implementation with preservation tests |

## Behavioural tests

Module-specific tests in `tests/test_model_direction_selective_rgc.py` cover:

- independent exact-relaxation voltage parity,
- invalid optical drive state preservation,
- corrupted runtime-buffer state preservation,
- spike reset semantics for voltage and temporal buffers.

Existing gap-model compatibility tests still cover On/Off centre responses,
surround inhibition, temporal derivative behaviour, and constructor validation.
Go, Rust engine, Julia, and Rust safety checks cover the aligned polyglot
contract.

## Benchmark

Measured locally on 2026-05-31 after exact-relaxation hardening.

| Runtime | Benchmark | Median | Per step | Artefact |
|---------|-----------|-------:|---------:|----------|
| Python | 100k periodic flash steps x 5 repeats | 188.4 ms per 100k | 1.884 us | `benchmarks/results/local_i5_11600k_python_2026-05-31_direction_selective_rgc.json` |

## Example

```python
from sc_neurocore.neurons.models.direction_selective_rgc import DirectionSelectiveRGC

cell = DirectionSelectiveRGC.new_on()
spike = cell.step_rf(intensity=6.0, surround_mean=0.5)
print(spike, cell.v, cell._surround)
```

## Notes

The model intentionally stays lightweight: biological DSGC direction selectivity
involves asymmetric starburst amacrine inhibition, while this implementation
captures the computational derivative-plus-surround contract used by the codebase
for O(1) visual-event preprocessing.
