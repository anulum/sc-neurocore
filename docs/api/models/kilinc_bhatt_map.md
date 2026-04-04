# KilincBhattMapNeuron

**Module:** `engine/src/neurons/maps.rs`
**Reference:** Kilinc & Bhatt, 2023 (minimal adaptive threshold map)
**Family:** 2D discrete map with adaptive threshold
**State variables:** `x` (output), `theta` (adaptive threshold)

---

## Biological Context

A minimal sigmoid map with built-in spike frequency adaptation through a slow threshold variable. Designed for efficient hardware implementation while retaining biologically relevant dynamics (adaptation, threshold crossing).

Key features:
- **Adaptive threshold**: theta increases on each spike, raising the bar for subsequent spikes
- **Spike frequency adaptation**: progressive theta increase slows firing
- **Hardware-friendly**: simple arithmetic, no transcendental functions beyond sigmoid
- **Self-stabilising**: negative feedback from -x term prevents runaway

---

## Equations

$$x(n+1) = -x(n) + k \cdot \sigma(4(x(n) - \theta(n))) + I$$
$$\theta(n+1) = \beta \cdot \theta(n) + \gamma \cdot H(x(n) - \theta_{spike})$$

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/maps.rs` |
| PyO3 wrapper | Yes (state: x, theta) |
| NetworkRunner wired | `NeuronVariant::KilincBhattMap` |
| `create_neuron("KilincBhattMap")` | Yes |
| `supported_models()` | Includes "KilincBhattMap" |
| STRONG tests | 9 |
| Benchmark | `kilinc_bhatt_100k_steps`: **8.19 ms** (81.9 ns/step), i5-11600K |
