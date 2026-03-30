# Loihi2Neuron

**Module:** `sc_neurocore.neurons.models.loihi2`
**Reference:** Intel 2021
**Family:** Hardware (neuromorphic chip emulator)
**State variables:** `s1` (membrane), `s2` (synaptic), `s3` (adaptation)

## Equations

$$s_3 \leftarrow s_3 - s_3 / \tau_3$$
$$s_2 \leftarrow s_2 - s_2/\tau_2 + I + w_{23} s_3$$
$$s_1 \leftarrow s_1 - s_1/\tau_1 + w_{12} s_2 + w_{13} s_3$$

Spike: $s_1 \geq \theta \Rightarrow s_1 \to s_{1,reset}$, $s_3 \leftarrow s_3 + \Delta_{s_3}$.

All arithmetic is **integer** with division-based decay (bit-shift on hardware).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau1` | 10 | s1 decay divisor |
| `tau2` | 5 | s2 decay divisor |
| `tau3` | 50 | s3 decay divisor |
| `w12` | 1 | s2 → s1 coupling |
| `w13` | 0 | s3 → s1 coupling |
| `w23` | 0 | s3 → s2 coupling |
| `s1_threshold` | 1000 | Spike threshold |
| `s3_incr` | 10 | Spike-triggered s3 increment |

## Behaviour

- **Programmable:** Cross-coupling weights (w12, w13, w23) configure
  the neuron as CUBA, COBA, or Izhikevich-like on the same silicon.
- **Integer only:** Division-based decay → maps to Loihi 2 microcode.
- **3-state adaptation:** s3 accumulates on each spike and decays,
  producing spike-frequency adaptation.
- **Deterministic:** Fully deterministic integer computation.

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 13 | construction, step binary, silent, spikes, 3 states, adaptation s3, s3 decay, integer type, rate increase, w12 coupling, stability, reset, deterministic |
| Network | 1 | Population |
| Analysis | 1 | spike_count |
| **Total** | **15** | |
