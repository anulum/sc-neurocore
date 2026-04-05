# SmoothMuscleCell

**Module:** `engine/src/neurons/misc.rs`
**Reference:** Hirst & Edwards, J Physiol 531:567, 2001; Imtiaz et al., Biophys J 83:1877, 2002
**Family:** Visceral/vascular smooth muscle with Ca²⁺ oscillations
**State variables:** `v`, `d` (CaL activation), `f` (CaL inactivation), `ca` (cytosolic Ca²⁺), `ca_store` (ER/SR Ca²⁺)

---

## Biological Context

Smooth muscle cells lack fast Na⁺ channels. Depolarisation is driven by L-type Ca²⁺ channels, and repolarisation by BK (Ca²⁺-activated K⁺). Slow wave oscillations arise from intracellular Ca²⁺ dynamics:

- **IP3R release**: Ca²⁺-induced Ca²⁺ release from ER/SR stores via IP3 receptors
- **SERCA pump**: reuptake of Ca²⁺ back into stores
- **CaL + BK interplay**: membrane oscillations coupled to Ca²⁺ waves

Key features:
- No fast Na⁺ — depolarisation is CaL-dependent
- IP3-mediated Ca²⁺ release + SERCA reuptake (two-pool Ca²⁺ model)
- BK channels for repolarisation (voltage + Ca²⁺ dependent)
- Slow oscillations (~3-12 cycles/min for GI slow waves)

---

## Equations

$$C_m \frac{dV}{dt} = -(I_{CaL} + I_{BK} + I_L) + I_{ext}$$
$$\frac{d[Ca]}{dt} = \alpha \cdot I_{CaL} + J_{IP3R} - J_{SERCA} - \frac{[Ca]}{\tau_{Ca}}$$
$$\frac{d[Ca]_{store}}{dt} = J_{SERCA} - J_{IP3R}$$

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/misc.rs` |
| PyO3 wrapper | Yes (state: v, ca, ca_store) |
| NetworkRunner wired | `NeuronVariant::SmoothMuscle` |
| `create_neuron("SmoothMuscleCell")` | Yes |
| `supported_models()` | Includes "SmoothMuscleCell" |
| STRONG tests | 10 |
| Benchmark | `smooth_muscle_1k_steps`: **149.8 µs** (149.8 ns/step), i5-11600K |
