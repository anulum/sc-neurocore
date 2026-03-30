# ParallelSpikingNeuron (PSN)

**Module:** `sc_neurocore.neurons.models.psn`
**Reference:** 2024 (learned temporal filter)
**Family:** ML-optimised (convolution-based)
**State variables:** `buffer` (circular), `_ptr`

## Equations

$$\text{score} = \sum_{k=0}^{n-1} w_k \cdot x_k$$

Spike when score ≥ θ. Buffer cleared on spike.
Default kernel: uniform $w_k = 1/K$ (averaging filter).

## Behaviour

- **Learned kernel:** 1D convolution over circular buffer enables temporal pattern detection.
- **Buffer-clear reset:** After spike, all buffer entries zeroed → fresh accumulation.
- **Rate ∝ input:** At I=θ, spikes every kernel_size steps.
- **Custom kernels:** Non-uniform weights allow temporal selectivity.

## Test Coverage — 21 tests

Isolation (5), scoring (5), custom kernel (1), edge cases (6), network (2), analysis (2).
