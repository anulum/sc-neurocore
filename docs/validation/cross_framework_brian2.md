# Brian2 Cross-Framework Parity Check

Validates SC-NeuroCore LIF neuron dynamics against Brian2 (the de facto
standard for computational neuroscience simulations).

## Reference

Stimberg, M., Brette, R. & Goodman, D.F.M. (2019). Brian 2, an intuitive
and efficient neural simulator. eLife 8:e47314.

## Test 1: Single LIF (Exact Parity)

Identical LIF neuron with constant current in both simulators:
- ODE: dv/dt = (-v + 14mV) / 20ms
- Threshold: 10mV, Reset: 0mV
- dt = 0.1ms, duration = 500ms

| Metric | Brian2 | SC-NeuroCore | Match |
|--------|--------|-------------|-------|
| Spike count | 20 | 20 | Exact |
| Max timing diff | - | 0.000 ms | Exact |
| Mean ISI | 25.00 ms | 25.00 ms | 0.00% |

Both simulators produce identical output to floating-point precision.

## Test 2: Population with Poisson Input (Statistical Parity)

100 LIF neurons driven by independent Poisson inputs (500 Hz, 2mV weight):

| Metric | Brian2 | SC-NeuroCore | Ratio |
|--------|--------|-------------|-------|
| Total spikes | 3,456 | 3,524 | 1.02 |
| Mean rate | 69.1 Hz | 70.5 Hz | 1.02 |
| Wallclock time | 0.507 s | 0.069 s | **7.3x faster** |

Within 2% despite different Poisson RNG seeds. The 7.3x speedup
is from NumPy-vectorised population stepping vs Brian2's numpy codegen.

## Test 3: Network API (Partial)

The SC-NeuroCore Network API ran but the model registry name
`StochasticLIFNeuron` was not recognised. The population/projection
API needs the correct model class name from `neurons.models.__all__`.

## Summary

- Exact LIF parity with Brian2 (spike count, timing, ISI all match)
- Statistical parity on stochastic populations (2% difference)
- 7.3x speedup on population simulation (Python backend)

## Test Files

- `benchmarks/results/brian2_parity_results.json` -- measured data
- Kaggle kernel: `anulum/sc-neurocore-brian2-parity-v2`
