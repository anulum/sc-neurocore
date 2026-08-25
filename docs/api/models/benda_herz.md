# BendaHerzNeuron

**Module:** `sc_neurocore.neurons.models.benda_herz`

`BendaHerzNeuron` implements Benda and Herz (2003), equations (8) and (45),
using the paper's Figure 8 example: `f0(x)=60*sqrt(max(x,0))`, `gamma=0`, and
linear `A_inf(f)=0.1*f`. The adaptation and phase equations are integrated
simultaneously with candidate-first RK4. Frequency is in hertz, time is in
milliseconds, and a sampled phase crossing resets phase exactly to zero.

```python
from sc_neurocore.neurons.models.benda_herz import BendaHerzNeuron

neuron = BendaHerzNeuron()
event = neuron.step(4.0)
```

The model is deterministic. The former stochastic logistic/hazard recurrence
is preserved separately as `SCStochasticRateAdaptationNeuron`.

Source: Benda, J. and Herz, A. V. M. (2003), *A universal model for
spike-frequency adaptation*, Neural Computation 15, 2523-2564,
DOI `10.1162/089976603322385063`.

Five-runtime dispatch is available through
`sc_neurocore.accel.benda_herz.simulate_benda_herz`.
