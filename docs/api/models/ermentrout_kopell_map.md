# ErmentroutKopellMapNeuron

**Module:** `engine/src/neurons/maps.rs`
**Reference:** Ermentrout & Kopell, SIAM J Appl Math 46:233, 1986
**Family:** Canonical Type I (theta neuron) in discrete-time form
**State variables:** `theta` (phase variable, [0, 2pi))

---

## Biological Context

The Ermentrout-Kopell canonical model is the normal form for Type I (saddle-node on invariant circle, SNIC) excitability. Type I neurons can fire at arbitrarily low frequencies near threshold, unlike Type II neurons which have a minimum frequency.

Key features:
- **Type I excitability**: continuous f-I curve starting from zero frequency
- **Phase model**: theta wraps around [0, 2pi), spike at pi
- **Canonical form**: mathematically equivalent to any SNIC bifurcation neuron
- **Minimal**: single ODE in phase space

---

## Equations

$$\dot{\theta} = (1 - \cos\theta) + (1 + \cos\theta) \cdot I$$

Spike when $\theta$ crosses $\pi$.

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/maps.rs` |
| PyO3 wrapper | Yes (state: theta) |
| NetworkRunner wired | `NeuronVariant::ErmentroutKopellMap` |
| `create_neuron("ErmentroutKopellMap")` | Yes |
| `supported_models()` | Includes "ErmentroutKopellMap" |
| STRONG tests | 9 (fire, silent, Type I, theta wraps, negative, NaN, extreme, reset, performance) |
| Benchmark | `ermentrout_kopell_100k_steps`: **5.45 ms** (54.5 ns/step), i5-11600K |
