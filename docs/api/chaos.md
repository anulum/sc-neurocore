# Chaos

Chaotic random number generation for stochastic spiking and noise injection.

- `ChaoticRNG` — Logistic map RNG: `x_{n+1} = r * x_n * (1 - x_n)`. Provides deterministic chaos (non-linear, broadband spectrum) unlike linear PRNGs. Useful for noise injection, stochastic resonance experiments, and cryptographic bitstream generation.

Default parameters: `r=4.0` (fully chaotic regime), `x=0.5` (initial condition). At `r=4.0`, the logistic map produces uniformly distributed output on (0, 1).

```python
from sc_neurocore.chaos import ChaoticRNG

rng = ChaoticRNG(r=4.0, x=0.123)
samples = [rng.next() for _ in range(1000)]
```

::: sc_neurocore.chaos.rng
    options:
      show_root_heading: true
