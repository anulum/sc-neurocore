# IbarzTanakaMapNeuron

**Module:** `sc_neurocore.neurons.models.ibarz_tanaka_map`
**Reference:** Ibarz et al. 2007
**Family:** Map-based (piecewise-linear bursting)
**State variables:** `x` (fast, ≈voltage), `y` (slow, ≈adaptation)

## Equations

$$x_{n+1} = f(x_n) + y_n + I$$
$$y_{n+1} = y_n - \mu(x_n + 1) + \mu\sigma$$

$$f(x) = \begin{cases} \alpha/(1-x) & x \leq 0 \\ \alpha + \beta x & x > 0 \end{cases}$$

Spike: $x \geq x_\theta$, reset $x \to x_{reset}$.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 3.65 | Piecewise map amplitude |
| `beta` | 0.25 | Linear spiking slope |
| `mu` | 0.0005 | Slow time-scale |
| `sigma` | -1.6 | Slow variable target |
| `x_threshold` | 3.0 | Spike threshold |
| `x_reset` | -1.0 | Post-spike reset |

## Behaviour

- **Discrete map:** No ODE — iterative, computationally cheap.
- **Piecewise-linear:** f(x) has a singularity at x=1 (from left),
  producing sharp spike onset. Linear spiking phase above x=0.
- **Bursting:** Slow y variable (µ=0.0005) modulates burst-pause.
- **Deterministic:** Fully deterministic map.
- **Efficient:** Single evaluation per step — ideal for large networks.

## Infrastructure Pipeline

```
IbarzTanakaMapNeuron
├── step(current) → int {0,1}
├── Population: works
├── Verilog: division LUT + comparator, ~30 LUTs
└── Rust: supported via NeuronVariant
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 10 | construction, step binary, subthreshold, spikes, piecewise f, slow y, reset on spike, rate increase, stability, reset, deterministic |
| Network | 1 | Population |
| Analysis | 1 | spike_count |
| **Total** | **12** | |
