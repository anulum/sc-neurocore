# CourageNekorkinMapNeuron

**Module:** `sc_neurocore.neurons.models.courage_nekorkin_map`
**Reference:** Courbage, Nekorkin & Vdovin 2007
**Family:** Map-based (piecewise-linear)
**State variables:** `x` (fast), `y` (slow)

## Equations

$$x_{n+1} = f(x_n) + y_n + I + J$$
$$y_{n+1} = y_n - \beta(x_n + 1)$$
$$f(x) = \begin{cases} \alpha x & x < 0 \\ \alpha x/(1+\alpha x) & x \geq 0 \end{cases}$$

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 3.0 | Map slope / saturation |
| `beta` | 0.001 | Slow variable coupling |
| `j` | 0.1 | Intrinsic drive |
| `x_threshold` | 1.0 | Spike threshold |

## Behaviour

- **Piecewise-linear:** No transcendental functions. Very fast (2M steps/s).
- **Divergent default params:** Map escapes to ±∞ at default parameters.
  Clip fix applied (±1e6). Model is functional in network (Poisson input
  creates transient threshold crossings).
- **Lorenz-type:** Related to Lorenz attractor dynamics.

## Infrastructure Pipeline

```
CourageNekorkinMapNeuron
├── step(current) → int {0,1}
├── State clipped to ±1e6 (divergence guard)
├── Population, Network: PoissonInput(weight=0.5, rate=500Hz)
├── Verilog: trivially compilable (~10 LUTs, no exp)
└── Rust: supported
```

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | 2.0 Msteps/s | Not measured |

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | construction, step binary, state finite, piecewise function, reset |
| Network | 2 | Population, network spikes |
| Analysis | 1 | spike_count |
| **Total** | **8** | |

Bug #6 fixed: divergence to NaN → clip to ±1e6.
