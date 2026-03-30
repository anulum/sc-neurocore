# LoihiCUBANeuron

**Module:** `sc_neurocore.neurons.models.loihi_cuba`
**Reference:** Davies et al. 2018 (Intel)
**Family:** Hardware (neuromorphic chip emulator)
**State variables:** `v` (membrane, int), `u` (synaptic current, int)

## Equations

$$u \leftarrow u - u/\tau_u + I_{weighted}$$
$$v \leftarrow v - v/\tau_v + u$$

Spike: $v \geq \theta \Rightarrow v \to v_{reset}$.

All integer arithmetic with division-based decay.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau_v` | 10 | Membrane decay divisor |
| `tau_u` | 5 | Synaptic current decay divisor |
| `v_threshold` | 1000 | Spike threshold |
| `v_reset` | 0 | Post-spike reset |

## Behaviour

- **CUBA:** Current-based (no conductance reversal potentials).
- **Integer only:** Division-based decay maps to Loihi 1 microcode.
- **2-state:** Simpler than Loihi2Neuron (no s3 adaptation).
- **Deterministic:** Fully deterministic integer computation.

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 10 | construction, step binary, silent, spikes, u accumulation, u decay, integer type, rate increase, reset, deterministic |
| Network | 1 | Population |
| Analysis | 1 | spike_count |
| **Total** | **12** | |
