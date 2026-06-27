# Dashboard — CLI Simulation Monitor

Text-based dashboard for monitoring spiking network simulations in the terminal. Displays per-neuron firing rates with trend indicators and bar charts. Designed for quick visual feedback during development — not a replacement for proper analysis tools.

## SCDashboard

Maintains a rolling history (last 20 timesteps) of firing rates per neuron. On each `update()`, prints a formatted table with:

- Neuron index
- Current firing rate
- Trend indicator: `/ UP` (rate increasing), `\ DWN` (decreasing), `- STY` (stable, ±0.01)
- Bar chart (rate × 20 characters)

| Parameter | Meaning |
|-----------|---------|
| `n_neurons` | Number of neurons to monitor |

**Methods:**

- `update(firing_rates, step)` — Record rates and render dashboard
- History is auto-truncated to 20 entries per neuron

## Usage

```python
from sc_neurocore.dashboard.text_dashboard import SCDashboard
import numpy as np

dash = SCDashboard(n_neurons=4)
for step in range(100):
    rates = np.random.rand(4).tolist()
    dash.update(rates, step=step)
```

**Output:**

```
--- SC DASHBOARD | Step 42 ---
Neuron   | Rate     | Trend (Last 5)
----------------------------------------
#0       | 0.731    | / UP ||||||||||||||
#1       | 0.245    | \ DWN |||||
#2       | 0.500    | - STY ||||||||||
#3       | 0.892    | / UP ||||||||||||||||||
----------------------------------------
```

## Verification

The dashboard surface is covered by the dedicated flat and package test paths:
`tests/test_dashboard.py` and `tests/test_dashboard/test_text_dashboard.py`.
The 2026-06-27 scoped verification passed Ruff, Ruff docstring checks, strict
mypy, `15 passed` focused dashboard tests, and 100% isolated coverage for
`sc_neurocore.dashboard.text_dashboard`.

The dashboard is a terminal rendering helper. Its Rust, Julia, and Mojo
counterpart files remain placeholder safety markers because this slice did not
change numerical behaviour or cross-language execution.

::: sc_neurocore.dashboard.text_dashboard
    options:
      show_root_heading: true
