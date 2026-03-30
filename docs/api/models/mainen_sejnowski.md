# MainenSejnowskiNeuron

**Module:** `sc_neurocore.neurons.models.mainen_sejnowski`
**Reference:** Mainen & Sejnowski 1996
**Family:** Conductance-based (2-compartment)
**State variables:** `vs` (soma V), `va` (axon V), `m`, `h`, `n` (axon gates)

## Equations

**Soma (passive):**
$$C_s \frac{dV_s}{dt} = -g_L(V_s - E_L) + \kappa(V_a - V_s) + I$$

**Axon (active):**
$$C_a \frac{dV_a}{dt} = -g_{Na}m^3h(V_a-E_{Na}) - g_K n(V_a-E_K) + \kappa(V_s-V_a)$$

20 sub-steps per call (dt_eff = 0.1 ms). Spike: upward crossing of V_θ=-20 mV at soma.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `g_na` | 3000.0 | Axon Na conductance (very high — fast initiation) |
| `g_k` | 1500.0 | Axon K conductance |
| `g_l` | 1.0 | Soma leak conductance |
| `kappa` | 10.0 | Soma↔axon coupling |
| `c_a` | 0.1 | Axon capacitance (small → fast) |
| `dt` | 0.005 | Sub-step size (ms) |

## Behaviour

- **Axon-initiated spikes:** Na channel in axon fires first; soma follows
  via coupling. Models the axon initial segment (AIS).
- **Single spike per episode:** With default params, axon Na saturates
  after one spike without external reset — structural limitation.
- **safe_exp guards:** exp arguments clipped to [-500, 500], gating
  variables clipped to [0, 1], voltages to [-200, 200].
- **20 sub-steps:** dt=0.005 ms × 20 = 0.1 ms effective per call.

## Known Limitations

- Default params produce only 1 spike then axon saturates at +200 mV.
  This is because the model lacks a post-spike reset mechanism for
  the axon compartment.

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 10 | construction, step binary, subthreshold, spike, 2 compartments, gating bounded, voltage clamped, stability, reset, deterministic |
| Network | 1 | Population |
| **Total** | **11** | |
