# EndocrineBetaCell

**Module:** `engine/src/neurons/misc.rs`
**Reference:** Chay & Keizer, Biophys J 42:181, 1983; Sherman et al., Biophys J 54:411, 1988
**Family:** Pancreatic beta cell with glucose-dependent bursting
**State variables:** `v`, `n` (K_dr activation), `ca` (intracellular Ca²⁺)

---

## Biological Context

Pancreatic beta cells in the islets of Langerhans secrete insulin in response to elevated blood glucose. The electrical signature is bursting: clusters of spikes on a slow wave, with burst duration encoding glucose concentration.

Key mechanisms:
- **No fast Na⁺**: depolarisation is entirely CaL-dependent
- **IK_ATP** (ATP-sensitive K⁺): the metabolic coupling — glucose → ATP → closes K_ATP → depolarisation
- **IK_Ca** (Ca²⁺-activated K⁺, SK): burst termination — Ca²⁺ accumulates during spike burst → SK activates → hyperpolarises → Ca²⁺ decays → next burst
- **ICaL** (L-type Ca²⁺): spike depolarisation during bursts
- **IK_dr** (delayed rectifier K⁺): spike repolarisation

Burst mechanism: Ca²⁺ is the slow variable. During a burst, Ca²⁺ rises → IK_Ca grows → eventually terminates the burst. During the silent phase, Ca²⁺ decays → IK_Ca weakens → next burst begins.

---

## Equations

$$C_m \frac{dV}{dt} = -(I_{CaL} + I_{K_{dr}} + I_{K_{ATP}} + I_{K_{Ca}} + I_L) + I_{ext}$$

$$I_{K_{ATP}} = g_{K_{ATP}} \cdot (1 - \text{ATP level}) \cdot (V - E_K)$$

$$I_{K_{Ca}} = g_{K_{Ca}} \cdot \frac{[Ca]^2}{[Ca]^2 + K_d^2} \cdot (V - E_K)$$

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `g_cal` | 5.0 mS/cm² | L-type Ca²⁺ |
| `g_kdr` | 4.0 mS/cm² | Delayed rectifier |
| `g_katp` | 3.0 mS/cm² | ATP-sensitive K (max) |
| `g_kca` | 2.0 mS/cm² | Ca²⁺-activated K (SK) |
| `atp_level` | 0.3 | ATP/ADP ratio (glucose proxy, 0-1) |
| `kd_kca` | 0.5 µM | SK Ca²⁺ Kd |
| `tau_ca` | 100 ms | Ca²⁺ decay |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/misc.rs` |
| PyO3 wrapper | Yes (state: v, n, ca) |
| NetworkRunner wired | `NeuronVariant::BetaCell` |
| `create_neuron("EndocrineBetaCell")` | Yes |
| `supported_models()` | Includes "EndocrineBetaCell" |
| STRONG tests | 10 |
| Benchmark | `beta_cell_1k_steps`: **185.0 µs** (185.0 ns/step), i5-11600K |
